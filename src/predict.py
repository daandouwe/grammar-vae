import torch
from torch.autograd import Variable

from grammar import gram
from util import load_data

# Location of saved model
model_path = '../checkpoints/model.pt'

# Load data
data_path = '../data/eq2_grammar_dataset.h5'
data = load_data(data_path)
gram = gram.split('\n')

x = data[0]
y = x.argmax(-1)
pred = [gram[i] for i in y]

model = torch.load(model_path)

x = Variable(torch.from_numpy(x).float().unsqueeze(0).transpose(-2, -1))
logits = model(x)
_, y_ = logits.unsqueeze(0).max(-1)

print('\n'.join(pred))
print(logits)
print(y_)
print(y)
