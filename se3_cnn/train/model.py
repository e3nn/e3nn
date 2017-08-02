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

    def load_files(self, files):
        """
        Returns what will be feeded to get_cnn()
        """
        raise NotImplementedError

    def evaluate(self, input):
        return self.get_cnn()(input)
