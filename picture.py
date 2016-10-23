#!/usr/bin/env python2
from glob import glob
from collections import defaultdict
from tensorflow.python.summary.event_accumulator import EventAccumulator
import math, argparse, re
import numpy as np
import matplotlib.pyplot as plt
from utils import *

parser = argparse.ArgumentParser(description='Average plotter')
parser.add_argument("ATTRIBUTE", type=str, default = 'average_loss')
args = parser.parse_args()


time = []
for i in range(500):
  time.append(defaultdict(list))

for path in glob('summaries/*/*'):
  type = path.split('/')[1]
  event_accumulator = EventAccumulator(path)
  event_accumulator.Reload()
  costs = map(lambda e: e.value, event_accumulator.Scalars(args.ATTRIBUTE))
  len_costs = len(costs) - 1
  for i, cost in enumerate(costs[:-1]):
    min_time = int(math.floor(float(i) / len_costs * len(time)))
    max_time = int(math.floor(float(i+1) / len_costs * len(time)))
    max_time = max(min_time + 1, max_time) # at least one index
    next_cost = costs[i+1]
    time_span = max_time - min_time - 1
    for ti in range(min_time, max_time):
      if time_span:
        alpha = (ti - min_time) / float(time_span)
      else:
        alpha = 1
      time[ti][type].append((1-alpha) * cost + alpha * next_cost)

x = list(range(len(time)))
median = defaultdict(list)
low = defaultdict(list)
high = defaultdict(list)

keys = set()

for step in time:
  for key, values in sorted(step.items()):
    keys.add(key)
    low[key].append(np.percentile(values, 25))
    median[key].append(np.median(values))
    high[key].append(np.percentile(values, 75))

keys = list(sorted(keys, key=natural_keys))

plt.rc('font', family='Droid Serif', weight='light')
plt.grid()
for key in keys:
  plt.fill_between(x, low[key], high[key], alpha=.3, linewidth=0, label=key)

for key in keys:
  plt.plot(x, median[key], label=key)

plt.legend(loc='upper right', fancybox=True, shadow=True)

plt.savefig(args.ATTRIBUTE + '.png', dpi=200)
#plt.plot()
