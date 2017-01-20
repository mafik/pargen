from __future__ import print_function
import sys, heapq, operator
from time import clock
from datetime import timedelta
import numpy as np
import contextlib
import os, re
import tensorflow as tf

def add_arguments(parser, model_name_required=True):
  parser.add_argument("--model-name", type=str, required=model_name_required, default=None)
  parser.add_argument("--params", type=str, default="best")
  if model_name_required:
    parser.add_argument("--restore", type=bool, default=True)
  else:
    restore_parser = parser.add_mutually_exclusive_group(required=True)
    restore_parser.add_argument('--restore', dest='restore', action='store_true')
    restore_parser.add_argument('--no-restore', dest='restore', action='store_false')

class ModelSession:
  def __init__(self, args):
    self.args = args
    self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

  def __enter__(self):
    self.session = tf.Session()
    self.session.as_default()
    self.session.run(tf.global_variables_initializer())
    if self.args.model_name:
      self.dir = "models/{}/".format(self.args.model_name)
      self.summary_writer = tf.summary.FileWriter(self.dir, self.session.graph)
      makedir(self.dir)
      if self.args.restore:
        self.saver.restore(self.session, self.dir + self.args.params + "_params")
    else:
      self.dir = ""
    return self

  def save_best(self, loss):
    if self.dir:
      self.saver.save(self.session, self.dir + "best_params")
      print(loss, file=open(self.dir + "best_loss.txt", 'w'))

  def save_last(self, loss):
    if self.dir:
      self.saver.save(self.session, self.dir + "last_params")
      print(loss, file=open(self.dir + "last_loss.txt", 'w'))

  def write_summary(self, summary, step):
    if self.dir:
      self.summary_writer.add_summary(summary, step)


  def __exit__(self, exc_type, exc_val, exc_tb):
    self.session.close()

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
  t = clock()
  write(("{}: " + format_string).format(timedelta(seconds=t - start_time), *args, **kwargs))
  return t

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

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def memoize(function):
  memo = {}
  def wrapper(*args):
    if args in memo:
      return memo[args]
    else:
      rv = function(*args)
      memo[args] = rv
      return rv
  return wrapper
