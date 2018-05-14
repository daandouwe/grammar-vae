import torch
from torch.autograd import Variable
from nltk import CFG, Nonterminal

grammar = """S -> S '+' T
S -> S '*' T
S -> S '/' T
S -> T
T -> '(' S ')'
T -> 'sin(' S ')'
T -> 'exp(' S ')'
T -> 'x'
T -> '1'
T -> '2'
T -> '3'
Nothing -> None"""

GCFG = CFG.fromstring(gram)

S, T = Nonterminal('S'), Nonterminal('T')

def get_mask(nonterminal, grammar, as_variable=False):
    if isinstance(nonterminal, Nonterminal):
        mask = [rule.lhs() == nonterminal for rule in grammar.productions()]
        mask = Variable(torch.FloatTensor(mask)) if as_variable else mask
        return mask
    else:
        raise ValueError('Input must be instance of nltk.Nonterminal')

if __name__ == '__main__':
    # Usage:
    GCFG = nltk.CFG.fromstring(grammar)

    print(get_mask(T))
