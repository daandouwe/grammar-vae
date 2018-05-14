import numpy as np
import torch
from torch.autograd import Variable

from model import GrammarVAE
from stack import Stack

from util import load_data, make_nltk_tree
from train import ENCODER_HIDDEN, Z_SIZE, DECODER_HIDDEN, OUTPUT_SIZE

torch.manual_seed(10)

# Load saved model
# model_path = '../checkpoints/model.pt'
# model = torch.load(model_path)
model = GrammarVAE(ENCODER_HIDDEN, Z_SIZE, DECODER_HIDDEN, OUTPUT_SIZE)

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

# def test():
x = data[0]
# y = x.argmax(-1)
# y_ = predict(x)
# a = evaluate(y, y_)

mu, sigma = model.encoder(data2input(x))
rules = model.generate(mu, max_length=5)
print(rules)

tree = make_nltk_tree(rules, 0)
tree.draw()
