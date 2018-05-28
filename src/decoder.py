import torch
import torch.nn as nn
from torch.autograd import Variable

from encoder import Encoder

class Decoder(nn.Module):
    """RNN decoder that reconstructs the sequence of rules from laten z"""
    def __init__(self, input_size, hidden_size, output_size, rnn_type='lstm'):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type

        self.linear_in = nn.Linear(input_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, output_size)

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        else:
            raise ValueError('Select rnn_type from [lstm, gru]')

        self.relu = nn.ReLU()

    def forward(self, z, max_length):
        """The forward pass used for training the Grammar VAE.

        For the rnn we follow the same convention as the official keras
        implementaion: the latent z is the input to the rnn at each timestep.
        See line 138 of
            https://github.com/mkusner/grammarVAE/blob/master/models/model_eq.py
        for reference.
        """
        x = self.linear_in(z)
        x = self.relu(x)

        # The input to the rnn is the same for each timestep: it is z.
        x = x.unsqueeze(1).expand(-1, max_length, -1)
        hx = Variable(torch.zeros(x.size(0), self.hidden_size))
        hx = (hx, hx) if self.rnn_type == 'lstm' else hx

        x, _ = self.rnn(x, hx)

        x = self.relu(x)
        x = self.linear_out(x)
        return x


if __name__ == '__main__':
    import h5py

    # First run the encoder
    Z_DIM = 2
    BATCH_SIZE = 100
    MAX_LENGTH = 15
    OUTPUT_SIZE = 12

    # Load data
    data_path = '../data/eq2_grammar_dataset.h5'
    f = h5py.File(data_path, 'r')
    data = f['data']

    # Create encoder
    encoder = Encoder(10, Z_DIM)

    # Pass through some data
    x = torch.from_numpy(data[:BATCH_SIZE]).transpose(-2, -1).float() # shape [batch, 12, 15]
    x = Variable(x)
    _, y = x.max(1) # The rule index


    mu, sigma = encoder(x)
    z = encoder.sample(mu, sigma)
    kl = encoder.kl(mu, sigma)

    decoder = Decoder(Z_DIM, 10, OUTPUT_SIZE)
    logits = decoder(z, max_length=MAX_LENGTH)

    criterion = torch.nn.CrossEntropyLoss()
    logits = logits.view(-1, logits.size(-1))
    y = y.view(-1)
    loss = criterion(logits, y)

    print(loss)
