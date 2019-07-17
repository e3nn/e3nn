# pylint: disable=C,R,E1101
'''
Architecture to predict molecule energy on database qm7

RMSE test = 5.7
'''
import torch
import torch.utils.data
from se3cnn.blocks.tensor_product import TensorProductBlock

import numpy as np
import scipy.io
import os
import time


class QM7(torch.utils.data.Dataset):
    url = 'http://quantum-machine.org/data/qm7.mat'
    mat_file = 'qm7.mat'

    def __init__(self, root, split, download=False):
        self.root = os.path.expanduser(root)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        data = scipy.io.loadmat(os.path.join(self.root, self.mat_file))
        indices = data['P'][split]  # shape = (5, 1433)
        self.positions = data['R'][indices]  # positions, shape = (7165, 23, 3)
        self.charges = data['Z'][indices].astype(np.int32)  # charge: 5 atom types: 1, 6, 7, 8, 16, shape = (7165, 23)
        self.energies = data['T'].flatten()[indices]  # labels: atomization energies in kcal/mol, shape = (7165, )

    def __getitem__(self, index):
        positions, charges, energy = self.positions[index], self.charges[index], self.energies[index]

        p = 0.3
        n = 64

        number_of_atoms_types = 5
        fields = np.zeros((number_of_atoms_types, n, n, n), dtype=np.float32)

        a = np.linspace(start=-n / 2 * p + p / 2, stop=n / 2 * p - p / 2, num=n, endpoint=True)
        xx, yy, zz = np.meshgrid(a, a, a, indexing="ij")

        for ch, pos in zip(charges, positions):
            if ch == 0:
                break

            ato = {1: 0, 6: 1, 7: 2, 8: 3, 16: 4}[ch]

            x = pos[0]
            y = pos[1]
            z = pos[2]

            density = np.exp(-((xx - x)**2 + (yy - y)**2 + (zz - z)**2) / (2 * p**2))
            density /= np.sum(density)

            fields[ato] += density

        return torch.FloatTensor(fields), energy

    def __len__(self):
        return len(self.energies)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.mat_file))

    def download(self):
        from six.moves import urllib

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(self.root)
        except OSError as e:
            if e.errno == os.errno.EEXIST:
                pass
            else:
                raise

        print('Downloading ' + self.url)
        data = urllib.request.urlopen(self.url)
        file_path = os.path.join(self.root, self.mat_file)
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print('Done!')


class CNN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        features = [
            (5, 0, 0),  # 64
            (10, 3, 0),  # 32
            (10, 3, 1),  # 32
            (16, 8, 1),  # 32
            (16, 8, 1),  # 32
            (16, 8, 1),  # 32
            (1, 0, 0)  # 32
        ]


        from se3cnn import basis_kernels
        radial_window_dict = {'radial_window_fct':basis_kernels.gaussian_window_fct_convenience_wrapper,
                              'radial_window_fct_kwargs':{'mode':'sfcnn', 'border_dist':0., 'sigma':.6}}
        common_block_params = {'size': 7, 'padding': 3, 'batch_norm_momentum': 0.01, 'batch_norm_mode': 'maximum', 'radial_window_dict':radial_window_dict}



        block_params = [
            {'stride': 2, 'activation': torch.nn.functional.relu},
            {'stride': 1, 'activation': torch.nn.functional.relu},
            {'stride': 1, 'activation': torch.nn.functional.relu},
            {'stride': 1, 'activation': torch.nn.functional.relu},
            {'stride': 1, 'activation': torch.nn.functional.relu},
            {'stride': 1, 'activation': None},
        ]

        assert len(block_params) + 1 == len(features)

        # I used TensorProductBlock because I did it before Taco proposed the HighwayBlock
        blocks = [TensorProductBlock(features[i], features[i + 1], **common_block_params, **block_params[i]) for i in range(len(block_params))]
        self.blocks = torch.nn.Sequential(*blocks)

        # This is a pretrained Perceptron that takes as input only the number of each atom types
        # I itself makes a RMSE of 20
        # The idea is that SE3Net will add a correction depending on the geometry of the molecule
        self.lin = torch.nn.Linear(5, 1)
        self.lin.weight.data[0, 0] = -69.14
        self.lin.weight.data[0, 1] = -153.3
        self.lin.weight.data[0, 2] = -99.04
        self.lin.weight.data[0, 3] = -97.76
        self.lin.weight.data[0, 4] = -80.44

        self.alpha = torch.nn.Parameter(torch.ones(1))

    def forward(self, inp):  # pylint: disable=W
        '''
        :param inp: [batch, features, x, y, z]
        '''
        x = self.blocks(inp)

        x = x.view(x.size(0), x.size(1), -1)  # [batch, features, x*y*z]
        x = x.mean(-1)  # [batch, features]

        x = x * self.alpha * 5

        inp = inp.view(inp.size(0), inp.size(1), -1).sum(-1)

        y = self.lin(inp)

        # output the sum of the Perceptron and the SE3Net
        return x + y


def main():
    torch.backends.cudnn.benchmark = True

    train_set = torch.utils.data.ConcatDataset([QM7('qm7', split=i, download=True) for i in range(4)])
    test_set = QM7('qm7', split=4)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    model = CNN()
    if torch.cuda.is_available():
        model.cuda()
    print("The model contains {} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    def train_step(data, target):
        model.train()
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)
        optimizer.zero_grad()

        prediction = model(data)
        loss = torch.nn.functional.mse_loss(prediction, target)

        loss.backward()
        optimizer.step()

        return loss.data[0]

    for epoch in range(7):
        total_mse = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            time_start = time.perf_counter()
            mse = train_step(data, target)

            total_mse += mse

            print("[{}:{}/{}] RMSE={:.2} <RMSE>={:.2} time={:.2}".format(
                epoch, batch_idx, len(train_loader), mse ** 0.5, (total_mse / (batch_idx + 1)) ** 0.5, time.perf_counter() - time_start))

    model.eval()
    se = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = torch.autograd.Variable(data, volatile=True), torch.autograd.Variable(target)
        output = model(data)
        se += torch.nn.functional.mse_loss(output, target, size_average=False).data[0]  # sum up batch loss
        print("{}/{}".format(batch_idx, len(test_loader)))

    mse = se / len(test_loader.dataset)
    rmse = mse ** 0.5

    print('TEST RMSE={}'.format(rmse))


if __name__ == '__main__':
    main()
