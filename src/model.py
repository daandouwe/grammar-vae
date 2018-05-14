import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Normal

from nltk import Nonterminal

from encoder import Encoder
from decoder import Decoder
from stack import Stack,
from grammar import GCFG, S, T, get_mask

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

    def forward(self, x, max_length=15):
        mu, sigma = self.encoder(x)
        z = self.sample(mu, sigma)
        logits = self.decoder(z, max_length=max_length)
        return logits

    def generate(self, z, sample=False, max_length=15):
        """Generate a valid expression from z using the decoder"""
        stack = Stack(grammar=GCFG, start_symbol=S)
        logits = self.decoder(z, max_length=max_length).squeeze()
        rules = []
        t = 0
        while not stack.empty:
            alpha = stack.pop()
            mask = get_mask(alpha, stack.grammar, as_variable=True)
            probs = mask * logits[t].exp()
            probs = probs / probs.sum()
            if sample:
                raise NotImplementedError
            else: # argmax
                _, i = probs.max(-1)
            # convert PyTorch Variable to regular integer
            i = i.data.numpy()[0]
            rule = stack.grammar.productions()[i]
            for symbol in reversed(rule.rhs()):
                if isinstance(symbol, Nonterminal):
                    stack.push(symbol)
            rules.append(rule)

            t += 1
            if t == max_length:
                break
        # if len(rules) < 15:
        #     pad = [stack.grammar.productions()[-1]]
        #     rules += (15 - len(rules))*pad

        return rules
