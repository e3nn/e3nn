# pylint: disable=C,R,E1101
'''
Minimalist example of usage of SE(3) CNN
'''
import torch
import numpy as np

from se3_cnn.blocks.highway import HighwayBlock


class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.features = [
            (1, 0, 0),
            (10, 10, 2),
            (2, 0, 0)
        ]
        self.block_params = [
            {'stride': 2, 'non_linearities': True},
            {'stride': 1, 'non_linearities': False},
        ]

        assert len(self.block_params) + 1 == len(self.features)

        for i in range(len(self.block_params)):
            block = HighwayBlock(self.features[i], self.features[i + 1], **self.block_params[i])
            setattr(self, 'block{}'.format(i), block)

    def forward(self, inp): # pylint: disable=W
        '''
        :param inp: [batch, features, x, y, z]
        '''
        x = inp

        for i in range(len(self.block_params)):
            block = getattr(self, 'block{}'.format(i))
            x = block(x)

        x = x.view(x.size(0), x.size(1), -1) # [batch, features, x*y*z]
        x = x.mean(-1) # [batch, features]

        return x


def main():
    cnn = CNN()
    cnn.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters())


    batch_size = 64
    sample_size = 32

    mesh = np.linspace(-1, 1, sample_size)
    mx, my, mz = np.meshgrid(mesh, mesh, mesh)

    def step(i):
        x = 0.1 * np.random.randn(batch_size, 1, sample_size, sample_size, sample_size)
        y = np.random.randint(0, 2, size=(batch_size,))

        for j, label in enumerate(y):
            rx = 0.7
            ry = 0.7
            rz = 0.7 if label == 0 else 0.2

            ellipsoid = (mx / rx)**2 + (my / ry)**2 + (mz / rz)**2 < 1

            x[j, 0, ellipsoid] += np.random.randint(2) * 2 - 1

        x = torch.FloatTensor(x).cuda()
        y = torch.LongTensor(y).cuda()

        x = torch.autograd.Variable(x)
        y = torch.autograd.Variable(y)

        optimizer.zero_grad()
        out = cnn(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        loss = loss.data.cpu().numpy()
        out = out.data.cpu().numpy()
        y = y.data.cpu().numpy()

        acc = np.sum(out.argmax(-1) == y) / batch_size

        print("{}: acc={}% loss={}".format(i, 100 * acc, float(loss)))

    for i in range(1000):
        step(i)

if __name__ == '__main__':
    main()
