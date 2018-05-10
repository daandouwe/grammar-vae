import torch
import torch.nn as nn
from torch.autograd import Variable

class Encoder(nn.Module):
    """Convolutional encoder for Grammar VAE. Applies a series of
    one-dimensional convolutions to a batch of one-hot encodings of
    a sequence"""
    def __init__(self, hidden_dim, z_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(15, 2, kernel_size=2)
        self.conv2 = nn.Conv1d(2, 3, kernel_size=3)
        self.conv3 = nn.Conv1d(3, 4, kernel_size=4)

        # self.linear = nn.Linear(conv_dim, hidden_dim)
        # self.mu = nn.Linear(hidden_dim)
        # self.sigma = nn.Linear(hidden_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        out = x.view(x.size(0), -1) # flatten
        return out

        # filters, kernel_size,
        # 'conv1': 2, 'conv2': 3, 'conv3': 4
        # h = Convolution1D(self.hypers['conv1'], self.hypers['conv1'], activation = 'relu', name='conv_1')(x)
        # h = BatchNormalization(name='batch_1')(h)
        # h = Convolution1D(self.hypers['conv2'], self.hypers['conv2'], activation = 'relu', name='conv_2')(h)
        # h = BatchNormalization(name='batch_2')(h)
        # h = Convolution1D(self.hypers['conv3'], self.hypers['conv3'], activation = 'relu', name='conv_3')(h)
        # h = BatchNormalization(name='batch_3')(h)
        # h = Flatten(name='flatten_1')(h)
        # h = Dense(self.hypers['dense'], activation = 'relu', name='dense_1')(h)
        #

if __name__ == '__main__':
    encoder = Encoder(10, 10)
    x = Variable(torch.ones((1, 15, 12)))
    print(x)
    y = encoder(x)
    print(y)
