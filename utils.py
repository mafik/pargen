import sys
import numpy as np

def write(s):
  sys.stdout.write("\r" + ' ' * 80)
  sys.stdout.write("\r" + s)
  sys.stdout.flush()

def weighted_choice(weights):
    totals = np.cumsum(weights)
    norm = totals[-1]
    throw = np.random.rand() * norm
    return np.searchsorted(totals, throw)

class Obj:
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      self.__dict__[k] = v