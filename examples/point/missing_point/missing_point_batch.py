"""
import torch
import torch.utils.data
import e3nn
from ase.db import connect
import pandas
import livelossplot as llp

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
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


# In[3]:


training_set_size = 1000
test_set_size = 10000

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
        self.N_max = np.max(self.N)
        self.coords = coords
        self.species = species
        self.species_list = species_list

    def __getitem__(self, index):
        N, _ = self.coords[index].shape

        species_onehot = np.zeros([self.N_max, len(self.species_list)])
        species_onehot[range(N),
                       list(map(self.species_list.index, self.species[index]))] = 1.0

        coords = np.zeros([self.N_max, 3])
        coords[:N] = self.coords[index]

        mask = np.zeros(self.N_max)
        mask[:N] = 1.0

        return (coords, species_onehot, mask)

    def __len__(self):
        return len(self.coords)


# In[6]:


class MissingPointNet(torch.nn.Module):
    def __init__(self, num_classes, num_radial=4, max_radius=2.5):
        super(MissingPointNet, self).__init__()

        features = [(num_classes,),
#                     (15,),
#                     (15, 15),
#                     (15, 15),
                    (24,),
                    (24, 24),
                    (24, 24),
                    [(1,), (0, 1), (num_classes,)]]
        self.num_features = len(features)

        nonlinearity = lambda x: torch.log(0.5 * torch.exp(x) + 0.5)

        sigma = np.sqrt(max_radius / num_radial / 2.)

        kwargs = {
            'radii': torch.linspace(0, max_radius, steps=num_radial, dtype=torch.float64),
            'activation': nonlinearity,
            'radial_function': partial(gaussian_radial_function, sigma=sigma)
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

    def forward(self, input, difference_mat, relative_mask):
        output = input # [B, N, C]
        output = self.layers[0](output) # [B, N, C]
        output = torch.transpose(output, -2, -1)  # [B, C, N]
        output = self.layers[1](output)
        output = self.layers[2](output, difference_mat, relative_mask)
        output = self.layers[3](output, difference_mat, relative_mask)
        prob = self.layers[4](output, difference_mat, relative_mask)
        coords = self.layers[5](output, difference_mat, relative_mask)
        atoms = self.layers[6](output, difference_mat, relative_mask)
        return prob, coords, atoms

    def __Rs_repr(self, features):
        return [(m, l) for l, m in enumerate(features)]

    def __capsule_dims(self, Rs):
        return [2 * n + 1 for mul, n in Rs for i in range(mul)]


# In[7]:


# IMPORTANT: L=1 Spherical harmonics are given in the canonical order (Y, Z, X)
# https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics


# In[8]:


def inference(net, shape, types, mask, remove_indices):
    '''
    net: network
    shape: tensor of shape [B, N_max, 3]
    types: tensor of shape [B, N_max, atom_types]
    mask: tensor of shape [B, N_max] indicating which atoms are present in an example
    remove_indices: List of shape [B, 1] Which atom is to be removed
    '''
    N_max = shape.shape[-2]
    batch = shape.shape[0]
    stay_indices = get_stay_indices(N_max, remove_indices)


    # Remove an atom
    new_shape = shape[np.arange(batch).reshape(-1, 1), stay_indices].to(device)  # [B, N-1, 3]
    new_types = types[np.arange(batch).reshape(-1, 1), stay_indices].to(device)  # [B, N-1, species_list]
    new_mask = mask[np.arange(batch).reshape(-1, 1), stay_indices].to(device) # [B, N-1]
#     print(new_shape.shape, new_types.shape, new_mask.shape)

    # Store removed atom
    removed_point = shape[np.arange(batch), remove_indices]  # [B, 3]
    removed_type = types[np.arange(batch), remove_indices]  # [B, species_list]
#     print(removed_point.shape, removed_type.shape)

    # Run the network
    relative_mask = e3nn.point_utils.relative_mask(new_mask)
    diff_mat = e3nn.point_utils.difference_matrix(new_shape)
    prob, coords, atoms = net(new_types, diff_mat, relative_mask)

    # Get rid of batch dim (NormSoftplus needed it)
    prob, coords, atoms = prob, coords, atoms

    # Make prob of masked atoms very negative
    prob[new_mask.unsqueeze(-2) < 1.0] = -1e6 # [B, 1, N]

    # Convert scalars to probabilities
    probability = torch.nn.functional.softmax(prob, dim=-1)

    # Get voted coords and atom type
    coords = torch.transpose(coords, -2, -1)  # [B, N-1, 3]
    coords = coords[..., [2, 0, 1]]  # [B, [Y, Z, X]] to [B, [X, Y, Z]]
    votes = coords + new_shape  # [B, N-1, 3]

    guess_coord = torch.einsum('bcn,bnd->bd', (probability, votes)).cpu()  # [B, 3]
    guess_atom = torch.einsum('bcn,bnd->bd',
                              (probability, torch.transpose(atoms, -2, -1))).cpu()  # [B, species_list]

    return {
        'new_shape': new_shape,
        'new_mask': new_mask,
        'removed_point': removed_point,
        'removed_type': removed_type,
        'guess_coord': guess_coord,
        'guess_atom': guess_atom,
        'votes': votes,
        'probability': probability,
    }


# In[9]:


def get_stay_indices(N_max, random_ints):
    indices = np.concatenate([[np.delete(np.arange(N_max),i)] for i in random_ints], axis=0)
    return indices

# get_stay_indices(10, [3,4,5,6,7])


# In[10]:


def train(net, dataloader, optimizer_state_dict=None, check_point_every=10):
    liveloss = llp.PlotLosses()
    net.train()

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=1e-2)

    if optimizer_state_dict:
        optimizer.load_state_dict(torch.load(optimizer_state_dict))

    max_epochs = 500

    for epoch in range(max_epochs):
        logs = {}
        epoch_loss = 0
        for data in dataloader:
            shape, types, mask = data
            N = np.sum(mask.detach().numpy(), axis=-1)
            remove_indices = [random.randrange(n) for n in N]

            result = inference(net, shape, types, mask, remove_indices)

            # Compute loss
            loss = loss_fn(result['guess_coord'], result['removed_point'])
            loss += loss_fn(result['guess_atom'], result['removed_type'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        logs['log loss'] = epoch_loss / len(dataloader)
        liveloss.update(logs)
        liveloss.draw()
#         print('Epoch {0}, Loss {1}'.format(epoch, epoch_loss / len(dataloader)))
        if epoch % check_point_every == 0:
            torch.save(net.state_dict(), './revised_batch_pth/missing_point_24c_epoch_{}.pth'.format(epoch))
            torch.save(optimizer.state_dict(), './revised_batch_pth/adam_missing_point_24c_epoch_{}.pth'.format(epoch))
    torch.save(net.state_dict(), './revised_batch_pth/missing_point_24c_epoch_{}.pth'.format(epoch))
    torch.save(optimizer.state_dict(), './revised_batch_pth/adam_missing_point_24c_epoch_{}.pth'.format(epoch))
    return net, liveloss


# In[11]:


batch_size = 64
qm9_train = QM9Dataset(qm9_coords, qm9_atoms, species_list)
dataloader = torch.utils.data.DataLoader(qm9_train, batch_size=batch_size, shuffle=True)
net = MissingPointNet(len(species_list)).to(device)
# net.load_state_dict(torch.load('./revised_batch_pth/missing_point_epoch_250.pth'))
# net.load_state_dict(torch.load('./revised_batch_pth/missing_point_1_epoch_250.pth'))


# In[12]:


optimizer_state_dict = None
# optimizer_state_dict = './revised_batch_pth/adam_missing_point_epoch_250.pth'
# optimizer_state_dict = './revised_batch_pth/adam_missing_point_1_epoch_250.pth'
train(net, dataloader, optimizer_state_dict)


# In[13]:


def test(net, dataloader):
    net.eval()

    guesses = []

    with torch.no_grad():
        loss_fn = torch.nn.MSELoss()

        epoch_loss = 0
        for data in dataloader:
            shape, types, mask = data
            N = np.sum(mask.detach().numpy(), axis=-1)
            remove_indices = [random.randrange(n) for n in N]

            result = inference(net, shape, types, mask, remove_indices)

            # Compute loss
            loss = loss_fn(result['guess_coord'], result['removed_point'])
            loss += loss_fn(result['guess_atom'], result['removed_type'])

            for key, value in result.items():
                try:
                    result[key] = value.cpu().detach().numpy()
                except:
                    pass

            guesses.append(result)

            epoch_loss += loss.item()
        print('Loss {0}'.format(epoch_loss / len(dataloader)))
    return guesses


# In[14]:


def test_one(net, dataloader, i_target=0):
    net.eval()

    guesses = []

    with torch.no_grad():
        loss_fn = torch.nn.MSELoss()

        epoch_loss = 0

        # Load
        shape, types, mask = dataloader.dataset.__getitem__(i_target)

        # Add batch
        shape, types, mask = (np.expand_dims(shape, axis=0),
                              np.expand_dims(types, axis=0),
                              np.expand_dims(mask, axis=0))

        # Convert to tensor
        shape, types, mask = (torch.from_numpy(shape),
                              torch.from_numpy(types),
                              torch.from_numpy(mask))

        N = np.sum(mask.detach().numpy(), axis=-1)
        remove_indices = [random.randrange(n) for n in N]

        result = inference(net, shape, types, mask, remove_indices)

        # Compute loss
        loss = loss_fn(result['guess_coord'], result['removed_point'])
        loss += loss_fn(result['guess_atom'], result['removed_type'])

        for key, value in result.items():
            try:
                result[key] = value.cpu().detach().numpy()
            except:
                pass

            epoch_loss += loss.item()
        print('Loss {0}'.format(epoch_loss / len(dataloader)))

        return result


# In[15]:


qm9_test = QM9Dataset(qm9_test_coords, qm9_test_atoms, species_list)
test_dataloader = torch.utils.data.DataLoader(qm9_test, batch_size=batch_size, shuffle=True)


# In[16]:


result = test_one(net, test_dataloader, i_target=1)


# In[17]:


import plotly
import plotly.offline
plotly.offline.init_notebook_mode(connected=False)
from visualize import visualize_missing_point
N = int(np.sum(result['new_mask']))
data = visualize_missing_point(
    result['new_shape'][0],
    result['removed_point'][0],
    result['guess_coord'][0],
    result['votes'][0][:N],
    result['probability'][0][0][:N],
    True)
plotly.offline.iplot(data)


# In[18]:


guesses = test(net, test_dataloader)


# In[19]:


for guess in guesses:
    for key, value in guess.items():
        batch = value.shape[0]
        guess[key] = np.split(guess[key], batch)


# In[20]:


df = pandas.concat([pandas.DataFrame.from_dict(guess) for guess in guesses],
                    ignore_index=True)


# In[21]:


# Argmax of atom type one hot encoding
df['guess_atom_argmax'] = df['guess_atom'].apply(np.argmax, axis=-1)
df['removed_atom_argmax'] = df['removed_type'].apply(np.argmax, axis=-1)

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


# In[22]:


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


# In[23]:


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


# In[24]:


acc_dist = 0.5  # We say if mae dist is less than 0.5 angstroms, it is accurate

print('Accurate type and coordinate')
print('\t{}\t{}'.format(
    len(df[(df['mae_dist'] < acc_dist) & df['correct_atom_type'] == True]),
    len(df)))

print('\nMAE of distance by atom type')
for i, atom_number in enumerate(species_list):
    atom_name = number_to_name[atom_number]
    print('{}\t{}\t{}'.format(atom_name,
                              len(df[(df['mae_dist'] < acc_dist) & \
                                     (df['correct_atom_type'] == True) & \
                                     (df['removed_atom_argmax'] == i)]),
                              len(df[df['removed_atom_argmax'] == i])))


# In[ ]:

"""
