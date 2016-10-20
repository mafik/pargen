import sys, heapq, operator
from time import clock
from datetime import timedelta
import numpy as np
import contextlib
import os

def write(s):
  sys.stdout.write("\r" + ' ' * 80)
  sys.stdout.write("\r" + s)
  sys.stdout.flush()

def weighted_choice(weights):
  totals = np.cumsum(weights)
  norm = totals[-1]
  throw = np.random.rand() * norm
  return np.searchsorted(totals, throw)

def top_k(weights, k):
  return zip(*heapq.nlargest(k, enumerate(weights), key=operator.itemgetter(1)))[0]

class Obj:
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      self.__dict__[k] = v

def chunks(l, n):
  """Yield successive n-sized chunks from l."""
  for i in range(0, len(l), n):
    yield l[i:i+n]

start_time = clock()
def log_start():
  global start_time
  start_time = clock()

def log(format_string, *args, **kwargs):
  write(("{}: " + format_string).format(timedelta(seconds=clock() - start_time), *args, **kwargs))

@contextlib.contextmanager
def status(text):
  log(text)
  pre_clock = clock()
  yield
  log(text + " DONE (" + str(clock() - pre_clock) + "s)\n")

def makedir(path):
  if not os.path.exists(path):
    os.makedirs(path)
  return path