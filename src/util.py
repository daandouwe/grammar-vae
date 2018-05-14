import time

import h5py

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
        return n*self.step
        # return max(1., n*self.step)

def load_data(data_path):
    """Returns the data as numpy array"""
    f = h5py.File(data_path, 'r')
    return f['data'][:]
