import h5py

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

        self.linear = nn.Linear(24, hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.sigma = nn.Linear(hidden_dim, z_dim)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, x):
        h = self.conv1(x)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.relu(h)
        h = self.conv3(h)
        h = self.relu(h)
        h = h.view(x.size(0), -1) # flatten

        h = self.linear(h)
        mu = self.mu(h)
        sigma = self.softplus(self.sigma(h))
        return mu, sigma

if __name__ == '__main__':
    # Load data
    data_path = '../data/eq2_grammar_dataset.h5'
    f = h5py.File(data_path, 'r')
    data = f['data']

    # Create encoder
    encoder = Encoder(10, 10)

    x = Variable(torch.from_numpy(data[0:100]).float())
    mu, sigma = encoder(x)

    print(x)
    print(mu)
