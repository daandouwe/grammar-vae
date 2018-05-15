#!/usr/bin/env python

import numpy as np
import torch
from torch.autograd import Variable

from model import GrammarVAE
from stack import Stack

from util import load_data, make_nltk_tree
from train import ENCODER_HIDDEN, Z_SIZE, DECODER_HIDDEN, OUTPUT_SIZE

torch.manual_seed(10)

# Load saved model
model_path = 'checkpoints/model.pt'
model = torch.load(model_path)
# model = GrammarVAE(ENCODER_HIDDEN, Z_SIZE, DECODER_HIDDEN, OUTPUT_SIZE)

# Load data
data_path = '../data/eq2_grammar_dataset.h5'
data = load_data(data_path)

def data2input(x):
    x = torch.from_numpy(x).float().unsqueeze(0).transpose(-2, -1)
    return Variable(x)

def predict(x):
    x = data2input(x)
    logits = model(x)
    _, y = logits.squeeze(0).max(-1)
    return y

def evaluate(y, y_):
    try:
        (y == y_).mean()
    except:
        y_ = y_.data.numpy()
    return (y == y_).mean()

x = data[0]
x = data2input(x)
mu, sigma = model.encoder(x)
rules = model.generate(mu, sample=False, max_length=15)
print(rules)

if len(rules) < 15:
    tree = make_nltk_tree(rules)
    print(tree)
    # tree.draw()
