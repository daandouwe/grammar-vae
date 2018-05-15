import time
from collections import defaultdict

import h5py
from nltk import Tree, Nonterminal

class Timer:
	"""A simple timer to use during training"""
	def __init__(self):
		self.time0 = time.time()

	def elapsed(self):
		time1 = time.time()
		elapsed = time1 - self.time0
		self.time0 = time1
		return elapsed

class AnnealKL:
	"""Anneal the KL for VAE based training"""
	def __init__(self, step=1e-3, rate=500):
		self.rate = rate
		self.step = step

	def alpha(self, update):
		n, _ = divmod(update, self.rate)
		return min(1., n*self.step)

def load_data(data_path):
	"""Returns the h5 dataset as numpy array"""
	f = h5py.File(data_path, 'r')
	return f['data'][:]

def make_nltk_tree(derivation):
	"""return a nltk Tree object based on the derivation (list or tuple of Rules)."""
	d = defaultdict(None, ((r.lhs(), r.rhs()) for r in derrivation))
	def make_tree(lhs, rhs):
		return Tree(lhs, (child if child not in d else make_tree(child) for child in d[lhs]))
		return Tree(lhs,
				(child if not isinstance(child, Nonterminal) else make_tree(child)
					for child in rhs))

	return make_tree(r.lhs(), r.rhs())
