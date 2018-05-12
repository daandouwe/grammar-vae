import torch
import torch.nn as nn
from torch.autograd import Variable

from encoder import Encoder
from decoder import Decoder

class GrammarVAE(nn.Module):
    def __init__(self, hidden_encoder_size, z_dim, hidden_decoder_size, output_size):
        super(GrammarVAE, self).__init__()
        self.encoder = Encoder(hidden_encoder_size, z_dim)
        self.decoder = Decoder(z_dim, hidden_decoder_size, output_size)
