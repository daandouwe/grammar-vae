import torch
import torch.nn as nn
from torch.autograd import Variable

from encoder import Encoder

class Decoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()

    def forward(self, input, max_length):
        hx = Variable(torch.zeros(input.size(0), self.hidden_size))
        # input is the input along all the steps!
        input = input.unsqueeze(1).expand(-1, max_length, -1)
        output, _ = self.rnn(input, (hx, hx))
        output = self.linear(self.relu(output))
        return output


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
