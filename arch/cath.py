# pylint: disable=C,R,E1101
'''
Architecture to predict the structural categories of proteins according to the CATH 
classification (www.cathdb.info) - at the Architecture level.

'''
import torch
import torch.utils.data
from se3_cnn.blocks import HighwayBlock

import numpy as np
import scipy.io
import os
import time
from timeit import default_timer as timer

class Cath(torch.utils.data.Dataset):
    url = 'https://github.com/deepfold/cath_datasets/blob/master/{}?raw=true'

    def __init__(self, dataset, split, download=False, use_density=True):
        self.root = os.path.expanduser("cath")

        if download:
            self.download(dataset)

        self.use_density = use_density
        
        if not self._check_exists(dataset):
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        data = np.load(os.path.join(self.root, dataset))
        split_start_indices = data['split_start_indices']
        split_range = list(zip(split_start_indices[0:], list(split_start_indices[1:])+[None]))[split]
        self.positions = data['positions'][split_range[0]:split_range[1]]
        self.atom_types = data['atom_types'][split_range[0]:split_range[1]]
        self.n_atoms = data['n_atoms'][split_range[0]:split_range[1]]
        self.labels = [tuple(v) if len(v)>1 else v[0] for v in data['labels'][split_range[0]:split_range[1]]]

        self.atom_type_set = np.unique(self.atom_types[0][:self.n_atoms[0]])
        self.n_atom_types = len(self.atom_type_set)
        self.atom_type_map = dict(zip(self.atom_type_set, range(len(self.atom_type_set))))

        self.label_set = sorted(list(set(self.labels)))
        self.label_map = dict(zip(self.label_set, range(len(self.label_set))))
        
    def __getitem__(self, index):

        time_stamp = timer()
        
        n_atoms    = self.n_atoms[index]
        positions  = self.positions[index][:n_atoms]
        atom_types = self.atom_types[index][:n_atoms]
        label      = self.label_map[self.labels[index]]

        p = 2.0
        n = 50

        if torch.cuda.is_available():
            fields = torch.cuda.FloatTensor(*(self.n_atom_types,)+(n,n,n)).fill_(0)
        else:
            fields = torch.zeros(*(self.n_atom_types,)+(n,n,n))
        
        if self.use_density:
        
            ## Numpy version ##
            # a = np.linspace(start=-n / 2 * p + p / 2, stop=n / 2 * p - p / 2, num=n, endpoint=True)
            # xx, yy, zz = np.meshgrid(a, a, a, indexing="ij")

            # fields_np = np.zeros((self.n_atom_types, n, n, n), dtype=np.float32)
            # for i, atom_type in enumerate(self.atom_type_set):

            #     # Extract positions with current atom type
            #     pos = positions[atom_types == atom_type]

            #     # Create grid x atom_pos grid
            #     posx_posx, xx_xx = np.meshgrid(pos[:,0], xx.reshape(-1))
            #     posy_posy, yy_yy = np.meshgrid(pos[:,1], yy.reshape(-1))
            #     posz_posz, zz_zz = np.meshgrid(pos[:,2], zz.reshape(-1))                            

            #     # Calculate density
            #     density = np.exp(-((xx_xx - posx_posx)**2 + (yy_yy - posy_posy)**2 + (zz_zz - posz_posz)**2) / (2 * (p)**2))

            #     # Normalize so each atom density sums to one
            #     density /= np.sum(density, axis=0)

            #     # Sum densities and reshape to original shape
            #     fields_np[i] = np.sum(density, axis=1).reshape(xx.shape)


            ## Pytorch version ##        

            # Create linearly spaced grid
            a = torch.linspace(start=-n / 2 * p + p / 2, end=n / 2 * p - p / 2, steps=n)
            if torch.cuda.is_available():
                a = a.cuda()

            # Pytorch does not suppoert meshgrid - do the repeats manually
            xx = a.view(-1,1,1).repeat(1, len(a), len(a))
            yy = a.view(1,-1,1).repeat(len(a), 1, len(a))
            zz = a.view(1,1,-1).repeat(len(a), len(a), 1)

            for i, atom_type in enumerate(self.atom_type_set):

                # Extract positions with current atom type
                pos = positions[atom_types == atom_type]

                # Transfer position vector to gpu
                pos = torch.FloatTensor(pos)
                if torch.cuda.is_available():
                    pos = pos.cuda()

                # Pytorch does not suppoert meshgrid - do the repeats manually
                # Numpy equivalent:
                # posx_posx, xx_xx = np.meshgrid(pos[:,0], xx.reshape(-1))
                # posy_posy, yy_yy = np.meshgrid(pos[:,1], yy.reshape(-1))
                # posz_posz, zz_zz = np.meshgrid(pos[:,2], zz.reshape(-1))                            
                xx_xx = xx.view(-1, 1).repeat(1, len(pos))
                posx_posx = pos[:,0].contiguous().view(1, -1).repeat(len(xx.view(-1)), 1)
                yy_yy = yy.view(-1, 1).repeat(1, len(pos))
                posy_posy = pos[:,1].contiguous().view(1, -1).repeat(len(yy.view(-1)), 1)
                zz_zz = zz.view(-1, 1).repeat(1, len(pos))
                posz_posz = pos[:,2].contiguous().view(1, -1).repeat(len(zz.view(-1)), 1)

                # Calculate density
                sigma = 0.5*p
                density = torch.exp(-((xx_xx - posx_posx)**2 + (yy_yy - posy_posy)**2 + (zz_zz - posz_posz)**2) / (2 * (sigma)**2))

                # Normalize so each atom density sums to one
                density /= torch.sum(density, dim=0)

                # Sum densities and reshape to original shape
                fields[i] = torch.sum(density, dim=1).view(xx.shape)
        else:

            for i, atom_type in enumerate(self.atom_type_set):
            
                # Extract positions with current atom type
                pos = positions[atom_types == atom_type]

                # Lookup indices and move to GPU
                indices = torch.LongTensor(np.ravel_multi_index(np.digitize(pos, a+p/2).T, dims=(n,n,n)))
                if torch.cuda.is_available():
                    indices = indices.cuda()

                # Set values
                fields[i].view(-1)[indices] = 1

            
        # assert((np.abs(fields.numpy() - fields_np)<0.001).all())


        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(xx.reshape(-1), yy.reshape(-1), zz.reshape(-1), s=10*fields.numpy().reshape(-1), c=fields.numpy().reshape(-1), cmap=plt.get_cmap("Blues"))
        # plt.show()
        # plt.savefig("grid.png")

        time_elapsed = timer() - time_stamp
        # print("Time spent on __getitem__: %.4f sec" % time_elapsed)
        
        return fields, label

    def __len__(self):
        return len(self.labels)

    def _check_exists(self, dataset):
        return os.path.exists(os.path.join(self.root, dataset))

    def download(self, dataset):
        from six.moves import urllib

        if self._check_exists(dataset):
            return

        # download files
        try:
            os.makedirs(self.root)
        except OSError as e:
            if e.errno == os.errno.EEXIST:
                pass
            else:
                raise

        print('Downloading ' + self.url.format(dataset))
        data = urllib.request.urlopen(self.url.format(dataset))
        file_path = os.path.join(self.root, dataset)
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print('Done!')

class AvgSpacial(torch.nn.Module):
    def forward(self, inp):  # pylint: disable=W
        # inp [batch, features, x, y, z]
        return inp.view(inp.size(0), inp.size(1), -1).mean(-1)  # [batch, features]


class BaseCNN(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inp):  # pylint: disable=W
        '''
        :param inp: [batch, features, x, y, z]
        '''
        x = self.sequence(inp)  # [batch, features]

        return x


class CNN(BaseCNN):

    def __init__(self, n_output):
        super().__init__()

        # The parameters of a HighwayBlock are:
        # - The representation multiplicities (scalar, vector and dim. 5 repr.) for the input and the output
        # - The stride, same as 2D convolution
        # - A parameter to tell if the non linearity is enabled or not (ReLU or nothing)
        features = [
            (1, ),  # As input we have a scalar field
            (4, 4, 4),  
            (4, 4, 4),
            (8, 8, 8),
            (8, 8, 8),
            (8, 8, 8),
            (128, )  
        ]
        common_block_params = {'n_radial': 1, 'batch_norm_before_conv': False}
        block_params = [
            {'activation': torch.nn.functional.relu, 'size': 5, 'stride': 2, 'padding': 0},
            {'activation': torch.nn.functional.relu, 'size': 5, 'stride': 1, 'padding': 0},
            {'activation': torch.nn.functional.relu, 'size': 5, 'stride': 1, 'padding': 0},
            {'activation': torch.nn.functional.relu, 'size': 5, 'stride': 1, 'padding': 0},
            {'activation': torch.nn.functional.relu, 'size': 3, 'stride': 1, 'padding': 0},
            {'activation': None, 'size':3},
        ]
        
        assert len(block_params) + 1 == len(features)

        blocks = [HighwayBlock(features[i], features[i + 1], **common_block_params, **block_params[i]) for i in range(len(block_params))]

        self.sequence = torch.nn.Sequential(
            *blocks,
            AvgSpacial(),
            torch.nn.Linear(128, n_output),
            torch.nn.BatchNorm1d(n_output),
            # torch.nn.ReLU(),
            # torch.nn.Linear(50, 10),
        )


class DeeperDense(BaseCNN):

    def __init__(self, n_output):
        super().__init__()

        # The parameters of a HighwayBlock are:
        # - The representation multiplicities (scalar, vector and dim. 5 repr.) for the input and the output
        # - The stride, same as 2D convolution
        # - A parameter to tell if the non linearity is enabled or not (ReLU or nothing)
        features = [
            (1,),  # As input we have a scalar field
            (4, 4, 4),
            (4, 4, 4),
            (8, 8, 8),
            (8, 8, 8),
            (8, 8, 8),
            (256,)
        ]
        common_block_params = {'n_radial': 1, 'batch_norm_before_conv': False}
        block_params = [
            {'activation': torch.nn.functional.relu, 'size': 5, 'stride': 2, 'padding': 0},
            {'activation': torch.nn.functional.relu, 'size': 5, 'stride': 1, 'padding': 0},
            {'activation': torch.nn.functional.relu, 'size': 5, 'stride': 1, 'padding': 0},
            {'activation': torch.nn.functional.relu, 'size': 5, 'stride': 1, 'padding': 0},
            {'activation': torch.nn.functional.relu, 'size': 3, 'stride': 1, 'padding': 0},
            {'activation': None, 'size': 3},
        ]

        assert len(block_params) + 1 == len(features)

        blocks = [HighwayBlock(features[i], features[i + 1], **common_block_params, **block_params[i]) for i in
                  range(len(block_params))]

        self.sequence = torch.nn.Sequential(
            *blocks,
            AvgSpacial(),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_output),
            torch.nn.BatchNorm1d(n_output)
        )


class DeeperConv(BaseCNN):

    def __init__(self, n_output):
        super().__init__()

        # The parameters of a HighwayBlock are:
        # - The representation multiplicities (scalar, vector and dim. 5 repr.) for the input and the output
        # - The stride, same as 2D convolution
        # - A parameter to tell if the non linearity is enabled or not (ReLU or nothing)
        features = [
            (1,),  # As input we have a scalar field
            (4, 4, 4),
            (4, 4, 4),
            (4, 4, 4),
            (4, 4, 4),
            (8, 8, 8),
            (8, 8, 8),
            (8, 8, 8),
            (128,)
        ]
        common_block_params = {'n_radial': 1, 'batch_norm_before_conv': False}
        block_params = [
            {'activation': torch.nn.functional.relu, 'size': 5, 'stride': 1, 'padding': 0},
            {'activation': torch.nn.functional.relu, 'size': 5, 'stride': 1, 'padding': 0},
            {'activation': torch.nn.functional.relu, 'size': 5, 'stride': 1, 'padding': 0},
            {'activation': torch.nn.functional.relu, 'size': 5, 'stride': 1, 'padding': 0},
            {'activation': torch.nn.functional.relu, 'size': 5, 'stride': 1, 'padding': 0},
            {'activation': torch.nn.functional.relu, 'size': 5, 'stride': 1, 'padding': 0},
            {'activation': torch.nn.functional.relu, 'size': 3, 'stride': 1, 'padding': 0},
            {'activation': None, 'size': 3},
        ]

        assert len(block_params) + 1 == len(features)

        blocks = [HighwayBlock(features[i], features[i + 1], **common_block_params, **block_params[i]) for i in
                  range(len(block_params))]

        self.sequence = torch.nn.Sequential(
            *blocks,
            AvgSpacial(),
            torch.nn.Linear(128, n_output),
            torch.nn.BatchNorm1d(n_output),
        )


model_classes = {"CNN": CNN,
                 "DeeperDense": DeeperDense,
                 "DeeperConv": DeeperConv}

def main(data_filename, model_class):

    # torch.backends.cudnn.benchmark = True

    train_set = torch.utils.data.ConcatDataset([Cath(data_filename, split=i, download=True) for i in range(7)])
    validation_set = Cath(data_filename, split=7)
    test_set = torch.utils.data.ConcatDataset([Cath(data_filename, split=i) for i in range(8,10)])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=30, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=30, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=30, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)

    n_output = len(validation_set.label_set)
    
    model = model_class(n_output = n_output)
    if torch.cuda.is_available():
        model.cuda()
    print("The model contains {} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001)

    for epoch in range(100):
        for batch_idx, (data, target) in enumerate(train_loader):
            time_start = time.perf_counter()

            target = torch.LongTensor(target)
            
            model.train()
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            x, y = torch.autograd.Variable(data), torch.autograd.Variable(target)

            # forward and backward propagation
            optimizer.zero_grad()
            out = model(x)
            loss = torch.nn.functional.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

            # download results on the CPU
            loss = loss.data.cpu().numpy()
            out = out.data.cpu().numpy()
            y = y.data.cpu().numpy()

            # compute the accuracy
            acc = np.sum(out.argmax(-1) == y) / len(y)

            print("[{}:{}/{}] loss={:.4} acc={:.2} time={:.2}".format(
                epoch, batch_idx, len(train_loader), float(loss), acc, time.perf_counter() - time_start))

        model.eval()
        loss_sum = 0
        outs = []
        ys = []
        for batch_idx, (data, target) in enumerate(validation_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            x, y = torch.autograd.Variable(data, volatile=True), torch.autograd.Variable(target)
            out = model(x)
            outs.append(out.data.cpu().numpy())
            ys.append(y.data.cpu().numpy())
            loss_sum += torch.nn.functional.cross_entropy(out, y, size_average=False).data[0]  # sum up batch loss
            # print("{}/{}".format(batch_idx, len(validation_loader)))

        out = np.concatenate(outs)
        y = np.concatenate(ys)

        # compute the accuracy
        acc = np.sum(out.argmax(-1) == y) / len(y)

        avg_loss = loss_sum / len(validation_loader.dataset)

        print('VALIDATION [{}:{}/{}] loss={:.4} acc={:.2}'.format(epoch, len(train_loader)-1, len(train_loader), avg_loss, acc))


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("data_filename",
                        help="The name of the data file (will automatically downloaded)")
    parser.add_argument("--model", choices=model_classes.keys(), default='CNN',
                        help="Which model definition to use (default: %(default)s)")
    args = parser.parse_args()

    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)
    
    main(data_filename=args.data_filename, model_class=model_classes[args.model])
