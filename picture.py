#!/usr/bin/env python2
from glob import glob
from collections import defaultdict
from tensorflow.python.summary.event_accumulator import EventAccumulator
import math, argparse, re, os, pickle
import numpy as np
import matplotlib.pyplot as plt
from utils import *

parser = argparse.ArgumentParser(description='Average plotter')
args = parser.parse_args()

buckets = 500

run_logs = defaultdict(list)

for path in glob('summaries/*/*.mrogalski'):
  pickle_path = os.path.splitext(path)[0] + '.pickle'
  if (not os.path.exists(pickle_path)) or (os.path.getmtime(path) >= os.path.getmtime(pickle_path)):
    print("Reading events from", path)
    histogram = tuple([] for i in range(buckets))
    event_accumulator = EventAccumulator(path)
    event_accumulator.Reload()
    tags = event_accumulator.Tags()['scalars']
    run_log = dict()
    for tag in tags:
      scalars = event_accumulator.Scalars(tag)
      tag_values = map(lambda e: e.value, scalars)
      histogram = tuple([] for i in range(buckets))
      tag_value_count = len(tag_values) - 1
      for i, cost in enumerate(tag_values[:-1]):
        min_bucket = int(math.floor(float(i) / tag_value_count * buckets))
        max_bucket = int(math.floor(float(i + 1) / tag_value_count * buckets))
        max_bucket = max(min_bucket + 1, max_bucket) # at least one index
        next_tag_value = tag_values[i + 1]
        bucket_span = max_bucket - min_bucket - 1
        for ti in range(min_bucket, max_bucket):
          if bucket_span:
            alpha = (ti - min_bucket) / float(bucket_span)
          else:
            alpha = 1
          histogram[ti].append((1-alpha) * cost + alpha * next_tag_value)
      min_step = min(e.step for e in scalars)
      max_step = max(e.step for e in scalars)
      run_log[tag] = (histogram, min_step, max_step)
    pickle.dump(run_log, open(pickle_path, 'w'))
  else:
    run_log = pickle.load(open(pickle_path))

  type = path.split('/')[1]
  run_logs[type].append(run_log)

x = list(range(buckets))
median = defaultdict(list)
low = defaultdict(list)
high = defaultdict(list)

keys = set()

for type, run_log_list in run_logs.items():
  print type
  for run_log in run_log_list:
    values, min, max = run_log['train_average_loss']
    print values, min, max
    keys.add(type)
    for v in values:
      median[type].append(v)

def smooth(series_dict):
  for key, l1 in series_dict.items():
    l2 = list(l1)
    for i in range(1, len(l1)-1):
      l2[i] = l1[i-1] * 0.25 + l1[i] * 0.5 + l1[i+1] * 0.25
    series_dict[key] = l2

smooth(low)
smooth(high)

keys = list(sorted(keys, key=natural_keys))

colors = 'red blue green brown black cyan magenta yellow'.split()
plt.rc('font', family='Droid Serif', weight='light')
plt.grid()

ceiling_tracker = []
for key, color in zip(keys, colors):
  ceiling_tracker.extend(median[key])
  print median[key]
  plt.plot(x[:len(median[key])], median[key], label=key, color=color)

import math

axes = plt.gca()
#axes.set_ylim([math.floor(np.min(ceiling_tracker) * 2) / 2, np.percentile(ceiling_tracker, 99.8)])
axes.set_ylim([0,2])
axes.set_xlim([0, buckets])
plt.legend(loc='upper right', fancybox=True, shadow=True)

plt.savefig('picture.png', dpi=200)
#plt.plot()
