import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from model import GrammarVAE
from util import Timer, AnnealKL, load_data

ENCODER_HIDDEN = 10
Z_SIZE = 2
DECODER_HIDDEN = 10
BATCH_SIZE = 100
MAX_LENGTH = 15
OUTPUT_SIZE = 12
LR = 1e-2
CLIP = 5.
PRINT_EVERY = 10

if __name__ == '__main__':
    # Load data
    data_path = '../data/eq2_grammar_dataset.h5'
    data = load_data(data_path)
    # Turn it into a float32 PyTorch Tensor
    data = torch.from_numpy(data).float()

    # Create model
    model = GrammarVAE(ENCODER_HIDDEN, Z_SIZE, DECODER_HIDDEN, OUTPUT_SIZE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    def batch_iter(data, batch_size):
        """A simple iterator over batches of data"""
        n = data.shape[0]
        i_prev = 0
        for i in range(0, n, batch_size):
            x = data[i:i+batch_size].transpose(-2, -1) # shape [batch, 12, 15]
            x = Variable(x)
            _, y = x.max(1) # The rule index
            yield x, y

    def train():
        batches = batch_iter(data, BATCH_SIZE)
        for step, (x, y) in enumerate(batches, 1):
            mu, sigma = model.encoder(x)
            z = model.sample(mu, sigma)
            logits = model.decoder(z, max_length=MAX_LENGTH)

            logits = logits.view(-1, logits.size(-1))
            y = y.view(-1)
            loss = criterion(logits, y)
            kl = model.kl(mu, sigma)

            alpha = anneal.alpha(step)
            elbo = loss + alpha*kl

            # Update parameters
            optimizer.zero_grad()
            elbo.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), CLIP)
            optimizer.step()

            # Logging info
            if step % PRINT_EVERY == 0:
                log['loss'].append(loss.data.numpy()[0])
                log['kl'].append(kl.data.numpy()[0])
                log['elbo'].append(elbo.data.numpy()[0])
                print(
                    '| step {}/{} | loss {:.4f} | kl {:.4f} |'
                    ' elbo {:.4f} | {:.0f} sents/sec |'.format(
                step, data.shape[0] // BATCH_SIZE,
                np.mean(log['loss'][-PRINT_EVERY:]),
                np.mean(log['kl'][-PRINT_EVERY:]),
                np.mean(log['elbo'][-PRINT_EVERY:]),
                BATCH_SIZE*PRINT_EVERY / timer.elapsed()
                    )
                )

    def save():
        torch.save(model, '../checkpoints/model.pt')

    timer = Timer()
    log = {'loss': [], 'kl': [], 'elbo': []}
    anneal = AnnealKL(step=1e-3, rate=100)

    train()

    save()
