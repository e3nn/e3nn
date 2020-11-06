import torch
import torch.utils.data
import e3nn
from ase.db import connect
import pandas

import sys, os
import random
import numpy as np

from e3nn.utils import torch_default_dtype
import e3nn.point_utils as point_utils
from e3nn.non_linearities import NormSoftplus
from e3nn.convolution import SE3PointConvolution
from e3nn.blocks.point_norm_block import PointNormBlock
from e3nn.point_kernel import gaussian_radial_function
from e3nn.SO3 import torch_default_dtype

from functools import partial

torch.set_default_dtype(torch.float64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[3]:


training_set_size = 1000
test_set_size = 100

with connect('qm9.db') as conn:
    qm9_coords = []
    qm9_atoms = []
    qm9_test_coords = []
    qm9_test_atoms = []
    for atoms in conn.select('4<natoms<=18', limit=training_set_size):
        qm9_coords.append(atoms.positions)
        qm9_atoms.append(atoms.numbers)
    for atoms in conn.select('natoms=19', limit=test_set_size):
        qm9_test_coords.append(atoms.positions)
        qm9_test_atoms.append(atoms.numbers)


# In[4]:


species_list = sorted(set(np.concatenate(qm9_atoms)).union(set(np.concatenate(qm9_test_atoms))))
print(species_list)

number_to_name = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
onehot_to_number = lambda x: species_list[x]


# In[5]:


class QM9Dataset(object):
    def __init__(self, coords, species, species_list=None):
        self.N = [len(coord) for coord in coords]
        self.coords = coords
        self.species = species
        self.species_list = species_list

    def __getitem__(self, index):
        N, _ = self.coords[index].shape
        species_onehot = np.zeros([N, len(self.species_list)])
        species_onehot[range(N),
                       list(map(self.species_list.index, self.species[index]))] = 1.0
        return (self.coords[index], species_onehot)

    def __len__(self):
        return len(self.coords)


# In[6]:


class MissingPointNet(torch.nn.Module):
    def __init__(self, num_classes, num_radial=4, max_radius=2.5):
        super(MissingPointNet, self).__init__()

        features = [(num_classes,),
                    (15,),
                    (15, 15),
                    (15, 15),
                    [(1,), (0, 1), (num_classes,)]]
        self.num_features = len(features)

        nonlinearity = lambda x: torch.log(0.5 * torch.exp(x) + 0.5)

        sigma = max_radius / num_radial

        kwargs = {
            'radii': torch.linspace(0, max_radius, steps=num_radial, dtype=torch.float64),
            'activation': nonlinearity,
            'radial_function': partial(gaussian_radial_function, sigma=3*sigma)
        }
        # Encoding layers
        self.layers = torch.nn.ModuleList([torch.nn.Linear(features[0][0], features[1][0]),
                                           NormSoftplus([1 for i in range(features[1][0])],
                                                         scalar_act=nonlinearity,
                                                         bias_min=0.5, bias_max=2.0)])
        # Convolutions
        self.layers.extend([PointNormBlock(features[i], features[i+1], **kwargs) for i in range(1, len(features) - 2)])
        # Final layers
        prob = SE3PointConvolution(self.__Rs_repr(features[-2]),
                                   self.__Rs_repr(features[-1][0]),
                                   kwargs['radii'],
                                   kwargs['radial_function'])
        coords = SE3PointConvolution(self.__Rs_repr(features[-2]),
                                     self.__Rs_repr(features[-1][1]),
                                     kwargs['radii'],
                                     kwargs['radial_function'])
        atoms = SE3PointConvolution(self.__Rs_repr(features[-2]),
                                    self.__Rs_repr(features[-1][2]),
                                    kwargs['radii'],
                                    kwargs['radial_function'])
        self.layers.extend([prob, coords, atoms])

    def forward(self, input, difference_mat):
        output = input # [B, N, C]
        output = self.layers[0](output) # [B, N, C]
        output = torch.transpose(output, -2, -1)  # [B, C, N]
        output = self.layers[1](output)
        output = self.layers[2](output, difference_mat)
        output = self.layers[3](output, difference_mat)
        prob = self.layers[4](output, difference_mat)
        coords = self.layers[5](output, difference_mat)
        atoms = self.layers[6](output, difference_mat)
        return prob, coords, atoms

    def __Rs_repr(self, features):
        return [(m, l) for l, m in enumerate(features)]

    def __capsule_dims(self, Rs):
        return [2 * n + 1 for mul, n in Rs for i in range(mul)]


# In[7]:


# IMPORTANT: L=1 Spherical harmonics are given in the canonical order (Y, Z, X)
# https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics


# In[8]:


def inference(net, shape, types, remove_index, N):
    list_N = list(range(N))
    stay_indices = list_N[:remove_index] + list_N[remove_index + 1:]

    # Remove an atom
    new_shape = shape[:, stay_indices].to(device)  # [1, N-1, 3]
    new_types = types[:, stay_indices].to(device)  # [1, N-1, species_list]

    # Store removed atom
    removed_point = shape[0, remove_index]  # [3]
    removed_type = types[0, remove_index]  # [species_list]

    # Run the network
    diff_mat = e3nn.point_utils.difference_matrix(new_shape)
    prob, coords, atoms = net(new_types, diff_mat)

    # Get rid of batch dim (NormSoftplus needed it)
    prob, coords, atoms = prob.squeeze(0), coords.squeeze(0), atoms.squeeze(0)

    # Convert scalars to probabilities
    probability = torch.nn.functional.softmax(prob, dim=-1)

    # Get voted coords and atom type
    coords = torch.transpose(coords, 0, 1)  # [N-1, 3]
    coords = coords[:, [2, 0, 1]]  # [Y, Z, X] to [X, Y, Z]
    votes = coords + new_shape.squeeze(0)  # [N-1, 3]

    guess_coord = torch.mm(probability, votes).squeeze().cpu()  # [3]
    guess_atom = torch.mm(probability, torch.transpose(atoms, 0, 1)).squeeze().cpu()  # [species_list]

    return {
        'new_shape': new_shape,
        'removed_point': removed_point,
        'removed_type': removed_type,
        'guess_coord': guess_coord,
        'guess_atom': guess_atom,
        'votes': votes,
        'probability': probability,
    }


# In[9]:


def train(net, dataloader, check_point_every=10):
    net.train()

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=1E-3)

    max_epochs = 100
    for epoch in range(max_epochs):
        epoch_loss = 0
        for data in dataloader:
            shape, types = data
            N = shape.shape[-2]
            remove_index = random.randrange(N)

            result = inference(net, shape, types, remove_index, N)

            # Compute loss
            loss = loss_fn(result['guess_coord'], result['removed_point'])
            loss += loss_fn(result['guess_atom'], result['removed_type'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print('Epoch {0}, Loss {1}'.format(epoch, epoch_loss / len(dataloader)))
        if epoch % check_point_every == 0:
            torch.save(net.state_dict(), './missing_point_epoch_{}.pth'.format(epoch))
    torch.save(net.state_dict(), './missing_point_epoch_{}.pth'.format(epoch))
    return net


# In[10]:


batch_size = 1
qm9_train = QM9Dataset(qm9_coords, qm9_atoms, species_list)
dataloader = torch.utils.data.DataLoader(qm9_train, batch_size=batch_size, shuffle=True)
net = MissingPointNet(len(species_list)).to(device)


# In[11]:


train(net, dataloader)


# In[23]:


def test(net, dataloader):
    net.eval()
    guesses = []

    with torch.no_grad():
        loss_fn = torch.nn.MSELoss()

        epoch_loss = 0
        for data in dataloader:
            shape, types = data
            N = shape.shape[-2]

            # For testing, we remove every atom in turn
            for remove_index in range(N):
                remove_index = random.randrange(N)

                result = inference(net, shape, types, remove_index, N)

                # Compute loss
                loss = loss_fn(result['guess_coord'], result['removed_point'])
                loss += loss_fn(result['guess_atom'], result['removed_type'])

                for key, value in result.items():
                    try:
                        result[key] = value.cpu().detach().numpy()
                    except:
                        pass

                result.update({'loss': loss.item()})
                result.update({'loss': loss.item()})

                guesses.append(result)

            epoch_loss += loss.item()
        print('Test, Loss {0}'.format(epoch_loss / len(dataloader)))
    return guesses


# In[24]:


def test_one(net, dataloader, i_target=0):
    net.eval()

    with torch.no_grad():
        result = None

        loss_fn = torch.nn.MSELoss()

        epoch_loss = 0

        # Load i_target
        shape, types = test_dataloader.dataset.__getitem__(i_target)

        # Add batch
        shape, types = np.expand_dims(shape, axis=0), np.expand_dims(types, axis=0)

        # Convert to tensor
        shape, types = torch.from_numpy(shape), torch.from_numpy(types)

        N = shape.shape[-2]
        remove_index = random.randrange(N)

        result = inference(net, shape, types, remove_index, N)

        # Compute loss
        loss = loss_fn(result['guess_coord'], result['removed_point'])
        loss += loss_fn(result['guess_atom'], result['removed_type'])

        for key, value in result.items():
            try:
                result[key] = value.cpu().detach().numpy()
            except:
                pass

        result.update({'loss': loss.item()})

        epoch_loss += loss.item()
        print('Test, Loss {0}'.format(epoch_loss / len(dataloader)))

        return result


# In[25]:


qm9_test = QM9Dataset(qm9_test_coords, qm9_test_atoms, species_list)
test_dataloader = torch.utils.data.DataLoader(qm9_test, batch_size=batch_size, shuffle=True)


# In[26]:


guesses = test(net, test_dataloader)


# In[27]:


df = pandas.DataFrame.from_dict(guesses)


# In[28]:


# Argmax of atom type one hot encoding
df['guess_atom_argmax'] = df['guess_atom'].apply(np.argmax, axis=0)
df['removed_atom_argmax'] = df['removed_type'].apply(np.argmax, axis=0)

# Indices by true atom type
element_indices = {value: None for key, value in number_to_name.items()}
for i, atom_number in enumerate(species_list):
    element = number_to_name[atom_number]
    element_indices[element] = df.index[df['removed_atom_argmax'] == i]

# Boolean of whether atom type predicted correctly
df['correct_atom_type'] = (df['guess_atom_argmax'] == df['removed_atom_argmax'])

# Distance between removed point and guessed point
df['mae_dist'] = (df['removed_point'] - df['guess_coord']).apply(
    np.linalg.norm, axis=-1)


# In[29]:


# Accuracy by atom type
print('Accuracy of atom type')
print('\t{}\t{}'.format(
    sum(df['correct_atom_type']),
    len(df['correct_atom_type'])
))

print('\nAccuracy of atom type by atom type')
for i, atom_number in enumerate(species_list):
    atom_name = number_to_name[atom_number]
    print('{}\t{}\t{}'.format(
        atom_name,
        sum(df['correct_atom_type'][element_indices[atom_name]]),
        len(element_indices[atom_name])
    ))


# In[30]:


print('MAE of distance')
print('\t{}\tAA'.format(
    "%.2f" % np.mean(df['mae_dist'])
))

print('\nMAE of distance by atom type')
for i, atom_number in enumerate(species_list):
    atom_name = number_to_name[atom_number]
    print('{}\t{}\tAA'.format(
        atom_name,
        "%.2f" % np.mean(df['mae_dist'][element_indices[atom_name]])
    ))


# In[31]:


acc_dist = 0.5  # We say if mae dist is less than 0.5 angstroms, it is accurate

print('Accurate type and coordinate')
print('\t{}\t{}'.format(
    len(df[(df['mae_dist'] < acc_dist) & df['correct_atom_type'] == True]),
    len(df)))

print('\nMAE of distance by atom type')
for i, atom_number in enumerate(species_list):
    print('\t{}\t{}'.format(len(df[(df['mae_dist'] < acc_dist) &                                    (df['correct_atom_type'] == True) &                                    (df['removed_atom_argmax'] == i)]),
                            len(df[df['removed_atom_argmax'] == i])))


# In[32]:


result = test_one(net, test_dataloader)


# In[33]:


import plotly
import plotly.offline
plotly.offline.init_notebook_mode(connected=False)
from visualize import visualize_missing_point
data = visualize_missing_point(
    result['new_shape'][0],
    result['removed_point'],
    result['guess_coord'],
    result['votes'],
    result['probability'][0],
    True)
plotly.offline.iplot(data)


# In[ ]:




