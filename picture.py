#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function
from glob import glob
from collections import defaultdict
from tensorflow.python.summary.event_accumulator import EventAccumulator
import math, pickle
import matplotlib.pyplot as plt
from utils import *

buckets = 500
smooth = lambda l: l[:1] + tuple(l[i - 1] * 0.25 + l[i] * 0.5 + l[i + 1] * 0.25 for i in range(1, len(l) - 1)) + l[-1:]
aggregates = dict()

<<<<<<< HEAD
plt.rc('font', family='Droid Serif', weight='light')
=======
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
>>>>>>> 31cae4dc3e25f89877382c5ecfba2ebfd702507a

'''
Aggregates a bunch of timeseries, usually related to the same property.
'''
class Aggregate:
  def __init__(self):
    self.histogram = tuple([] for i in range(buckets))
  def compute_statistics(self):
    self.x = tuple(i * self.right / buckets for i, values in enumerate(self.histogram) if values)
    self.low = tuple(np.percentile(values, 50 - 34) for values in self.histogram if values)
    self.median = tuple(np.median(values) for values in self.histogram if values)
    self.high = tuple(np.percentile(values, 50 + 34) for values in self.histogram if values)
    self.low = smooth(self.low)
    self.high = smooth(self.high)

log("Loading aggregates...\n")
for type_path in glob('summaries/*/'):
  type = type_path.split('/')[1]
  type_cache_path = type_path + "aggregate.pickle"
  dir_mtime = max(os.path.getmtime(path) for path in glob(type_path + '*.box'))
  if (not os.path.exists(type_cache_path)) or (dir_mtime >= os.path.getmtime(type_cache_path)):
    type_dict = dict()
    log("Computing aggregates for {}...\n".format(type))

    properties = set()
    run_logs = []
    for path in glob(type_path + '*.box'):
      pickle_path = os.path.splitext(path)[0] + '.pickle'
      if (not os.path.exists(pickle_path)) or (os.path.getmtime(path) >= os.path.getmtime(pickle_path)):
        print("Reading events from", path)
        event_accumulator = EventAccumulator(path)
        event_accumulator.Reload()
        tags = event_accumulator.Tags()['scalars']
        run_log = dict()
        plt.grid()
        xmin = 999999
        xmax = -999999
        for tag in tags:
          scalars = event_accumulator.Scalars(tag)
          run_log[tag] = scalars
          plt.plot([x.step for x in scalars], [x.value for x in scalars], label=tag)
          xmin = min(xmin, min(x.step for x in scalars))
          xmax = max(xmax, max(x.step for x in scalars))
        plt.gca().set_xlim([xmin, xmax])
        plt.legend(loc='upper right', fancybox=True, shadow=True)
        plt.savefig(os.path.splitext(path)[0] + '.png', dpi=200)
        plt.close()
        pickle.dump(run_log, open(pickle_path, 'w'))
      else:
        run_log = pickle.load(open(pickle_path))
      for property in run_log.keys():
        properties.add(property)
      run_logs.append(run_log)

    for property in properties:
      type_dict[property] = Aggregate()

      def steps():
        for run_log in run_logs:
          if property in run_log:
            for e in run_log[property]:
              yield e.step

      type_dict[property].left = min(steps())
      type_dict[property].right = max(steps())

    for run_log in run_logs:
      for property, timeseries in run_log.items():
        aggregate = type_dict[property]
        for i, scalar in enumerate(timeseries[:-1]):
          next_scalar = timeseries[i + 1]
          min_bucket = int(math.floor(float(scalar.step) / aggregate.right * buckets))
          max_bucket = int(math.floor(float(next_scalar.step) / aggregate.right * buckets))
          max_bucket = max(min_bucket + 1, max_bucket)  # at least one index
          for ti in range(min_bucket, max_bucket):
            alpha = (ti - min_bucket) / float(max_bucket - min_bucket)
            aggregate.histogram[ti].append((1 - alpha) * scalar.value + alpha * next_scalar.value)
    for aggregate in type_dict.values():
      aggregate.compute_statistics()
    pickle.dump(type_dict, open(type_cache_path, 'w'))
    aggregates[type] = type_dict
  else:
    aggregates[type] = pickle.load(open(type_cache_path))

log("Drawing graph...\n")
types = list(t for t in sorted(aggregates.keys(), key=natural_keys) if not t.startswith('tf-'))

colors = 'red blue green brown cyan magenta yellow black'.split()
plt.grid()

for type, color in zip(types, colors):
  if 'train_average_loss' in aggregates[type]:
    a = aggregates[type]['train_average_loss']
    plt.fill_between(a.x, a.low, a.high, alpha=.2, linewidth=0, color=color)

ceiling_tracker = []
for type, color in zip(types, colors):
  if 'train_average_loss' in aggregates[type]:
    a = aggregates[type]['train_average_loss']
    ceiling_tracker.extend(a.median)
    plt.plot(a.x, a.median, label=type, color=color)
  if 'test_average_loss' in aggregates[type]:
    a = aggregates[type]['test_average_loss']
    ceiling_tracker.extend(a.median)
    plt.plot(a.x, a.median, color=color, linestyle='--')

import math

axes = plt.gca()
floor = lambda x: math.floor(x * 5) / 5
#axes.set_ylim([floor(np.min(ceiling_tracker)), floor(np.percentile(ceiling_tracker, 98))])
#axes.set_ylim([np.min(ceiling_tracker) - 0.1, np.max(ceiling_tracker) + 0.1])
axes.set_ylim([1, 2])

plt.legend(loc='upper right', fancybox=True, shadow=True)

plt.savefig('results.png', dpi=150)

log("Done!\n")
#plt.plot()
