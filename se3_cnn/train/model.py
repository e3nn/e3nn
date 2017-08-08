#pylint: disable=C,R,E1101
import torch

class Model:
    def initialize(self, number_of_classes):
        pass

    def get_cnn(self):
        """
        Returns a torch.nn.Module
        """
        raise NotImplementedError

    def get_batch_size(self):
        raise NotImplementedError

    def get_learning_rate(self, epoch):
        raise NotImplementedError

    def get_optimizer(self):
        return torch.optim.Adam(self.get_cnn().parameters())

    def get_criterion(self):
        return torch.nn.CrossEntropyLoss()

    def load_train_files(self, files):
        return self.load_files(files)

    def load_eval_files(self, files):
        return self.load_files(files)

    def load_files(self, files):
        """
        Returns a torch.FloatTensor
        """
        raise NotImplementedError

    def evaluate(self, x):
        x = torch.autograd.Variable(x, volatile=True)
        y = self.get_cnn()(x)
        return y.data.cpu().numpy()
