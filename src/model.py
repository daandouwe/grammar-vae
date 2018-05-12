import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Normal

from encoder import Encoder
from decoder import Decoder

class GrammarVAE(nn.Module):
    
    def __init__(self, hidden_encoder_size, z_dim, hidden_decoder_size, output_size):
        super(GrammarVAE, self).__init__()
        self.encoder = Encoder(hidden_encoder_size, z_dim)
        self.decoder = Decoder(z_dim, hidden_decoder_size, output_size)

    def sample(self, mu, sigma):
        """Reparametrized sample from a N(mu, sigma) distribution"""
        normal = Normal(torch.zeros(mu.shape), torch.ones(sigma.shape))
        eps = Variable(normal.sample())
        z = mu + eps*torch.sqrt(sigma)
        return z

    def kl(self, mu, sigma):
        """KL divergence between two normal distributions"""
        return torch.mean(-0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp(), 1))
