#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function

from itertools import chain

import data, datetime, argparse, sys

import numpy as np
import tensorflow as tf

from tensorflow.python.ops.rnn_cell import *
from utils import *
from kneser_ney import KneserNeyLM
from nltk.util import ngrams as ngram_generator
from collections import Counter

parser = argparse.ArgumentParser(description='Neural language model')
parser.add_argument("-n", "--run-name", type=str, default=None)
parser.add_argument("-a", "--learning-rate", type=float, default=0.01)
parser.add_argument("-i", "--bootstrap-in", type=bool, default=False)
parser.add_argument("-o", "--bootstrap-out", type=bool, default=False)
parser.add_argument("-m", "--bootstrap-mem", type=bool, default=False)
args = parser.parse_args()

if args.run_name == 'date':
  args.run_name = datetime.datetime.now().strftime("%A %H%M")

with status("Reading NKJP..."):
  arrays, lengths, symbol_map, symbol_count = data.read_nkjp_simple()

with status("Loading n-gram model..."):
  n = 5
  ngram_db_path = "ngram_db_{}.npz".format(n)
  try:
    npzfile = np.load(ngram_db_path)
    ngram_index = npzfile["concatenated"]
    ngram_db = npzfile["db"]
  except IOError as e:
    log("Building n-gram model from scratch...")

    ngrams = tuple(Counter() for i in range(n+1))
    ngrams[n].update(ngram for array in arrays for ngram in ngram_generator(chain((data.GO,)*n, array, (data.EOS,)), n))
    for i in range(n - 1, 0, -1):
      for ngram, count in ngrams[i+1].items():
        ngrams[i][ngram[1:]] += count
    ngram_counts = tuple(sum(order.values()) for order in ngrams)

    # unique_prefixes[i][ngram] where len(ngram) == i contains number of different symbols that proceed given ngram
    unique_prefixes = tuple(Counter() for i in range(n))
    # unique_suffixes[i][ngram] where len(ngram) == i contains number of different symbols that follow given ngram
    unique_suffixes = tuple(Counter() for i in range(n))
    for i in range(n, 0, -1):
      unique_prefixes[i-1].update(ngram[1:] for ngram in ngrams[i].keys())
      unique_suffixes[i-1].update(ngram[:-1] for ngram in ngrams[i].keys())

    discount = (1.0, 0.5, 0.75, 0.75, 0.75, 0.75)

    @memoize
    def prob(full_ngram):
      current_p = 0.0
      p_multiplier = 1.0
      for i in range(n, 0, -1):
        ngram = full_ngram[-i:]
        prefix = ngram[:-1]
        estamation_base = ngrams if i == n else unique_prefixes
        p = max(0, estamation_base[i][ngram] - discount[i]) / estamation_base[i-1][prefix]
        current_p += p * p_multiplier
        p_multiplier *= discount[i] / estamation_base[i-1][prefix] * unique_suffixes[i-1][prefix]
      current_p += p_multiplier / symbol_count # probability of an unseen symbol
      #print(u"Prob of {}: {}".format(''.join(symbol_map[c] for c in ngram), prob_cache[ngram]))
      return current_p

    ngram_db = tuple(list() for i in range(n))
    ngram_idx = tuple(dict() for i in range(n))
    ngram_predictions = []
    n1 = n - 1
    for index, arr in enumerate(arrays):
      log("Precomputing ngram probabilities... {}/{}".format(index, len(arrays)))
      ngram_stack = np.zeros([n], dtype=np.int32)

      for i in range(n):
        ngram_length = i + 1
        for prefix in ngram_generator(chain((data.GO,) * ngram_length, arr), ngram_length):
          pass

      for prefix in ngram_generator(chain((data.GO,) * n1, arr), n1):
        if prefix not in ngram_idx[n1]:
          probs = np.array([prob(prefix + (i,)) for i in range(symbol_count)])
          probs = probs / np.sum(probs)
          ngram_idx[n1][prefix] = len(ngram_db)
          ngram_db.append(probs)
        pred_symbol = np.argmax(ngram_db[ngram_idx[n1][prefix]])
        #print(u"Prefix '{}', prediction {:3d} '{}'".format(''.join(symbol_map[c] for c in prefix), pred_symbol, symbol_map[pred_symbol]))
        ngram_predictions.append(ngram_idx[prefix])
      ngram_predictions.append(0)
    ngram_index = np.array(ngram_predictions)
    ngram_db = np.array(ngram_db)
    np.savez(ngram_db_path, concatenated=ngram_index, db=ngram_db)

# TODO: retrain ngrams on the test set
# TODO: use more lengths of n-grams

def translate1(array):
  return ''.join(symbol_map[i] for i in array)

def translate2(array):
  return '\n'.join(translate1(row) for row in array)

batch_size = 400
train_test_divider = int(len(lengths) * 0.8)
sequence_count = len(lengths)
test_count = sequence_count - train_test_divider
check_indices = True
unrolled_iterations = 300
network_size = 200
layer_count = 2
ngram_count = len(ngram_db)

concatenated_arrays = np.concatenate([np.concatenate(([data.GO], arr, [data.EOS])) for arr in arrays])

# [sum(len(a) + 2 for a in arrays)] : int32
dataset = tf.Variable(concatenated_arrays,
                      trainable=False, dtype=tf.int32, name="flat_dataset")

ngram_index = tf.Variable(ngram_index, trainable=False, dtype=tf.int32, name="ngram_index")
ngram_db = tf.Variable(ngram_db, trainable=False, dtype=tf.float32, name="ngram_db")

# [num_sequences] : int32
sequence_lengths = tf.Variable([l+2 for l in lengths], trainable=False, name="sequence_lengths")

# [num_sequences] : int32
sequence_offsets = tf.cumsum(sequence_lengths, exclusive=True, name="sequence_offsets")

ngram_embedding_size = 10
ngram_embeddings = tf.Variable(tf.random_normal([ngram_count, ngram_embedding_size], mean=0.0, stddev=0.02),
                               trainable=True, name="ngram_embeddings")

def make_cell(dropout):
  cell = LSTMCell(network_size, use_peepholes=True, cell_clip=10, state_is_tuple=True)
  if dropout:
    cell = DropoutWrapper(cell, output_keep_prob=.5)
  cell = MultiRNNCell([cell] * layer_count, state_is_tuple=True)
  cell = InputProjectionWrapper(cell, num_proj=network_size)
  cell = OutputProjectionWrapper(cell, output_size=symbol_count)
  return cell

def build_input_vector(input_sequence, ngram_index_sequence, ngram_sequence):
  input_list = []
  input_list.append(tf.one_hot(input_sequence, symbol_count))
  if args.bootstrap_in:
    mean, variance = tf.nn.moments(ngram_sequence, [2], keep_dims=True)
    input_list.append(tf.nn.batch_normalization(ngram_sequence, mean, variance, 0, 1, 0.00001))
  if args.bootstrap_mem:
    input_list.append(tf.gather(ngram_embeddings, ngram_index_sequence))
  return input_list[0] if len(input_list) == 1 else tf.concat(2, input_list)


def get_loss(input_sequence, ngram_sequence, outputs, expected_sequence, consumed_sequence_lengths):
  if args.bootstrap_out:
    outputs = tf.add(outputs, tf.log(ngram_sequence))
  # [batch_size, unrolled_iterations]
  losses = tf.nn.sparse_softmax_cross_entropy_with_logits(outputs, expected_sequence)
  losses = tf.select(tf.equal(input_sequence, data.EOS), tf.zeros_like(losses), losses)
  total_loss = tf.reduce_sum(losses)
  return total_loss, total_loss / tf.cast(tf.reduce_sum(consumed_sequence_lengths), dtype=tf.float32)

def build_batch(indices, length, progress=None):
  if progress == None:
    progress = tf.zeros_like(indices)
  lengths = tf.gather(sequence_lengths, indices, check_indices, name="lengths")
  start_offsets = tf.gather(sequence_offsets, indices, check_indices, name="start_offsets")
  current_offsets = tf.add(start_offsets, progress, name="current_offsets")
  remaining_lengths = tf.sub(lengths, progress, name="remaining_lengths")
  max_range = tf.sub(remaining_lengths, 1)
  range_matrix = tf.minimum(*tf.meshgrid(max_range, tf.range(length + 1), indexing='ij'), name="range_matrix")
  indices_to_gather = tf.add(tf.tile(tf.expand_dims(current_offsets, 1), [1, length + 1]),
                             range_matrix, name='indices_to_gather')
  sequence = tf.gather(dataset, indices_to_gather, check_indices, name="sequence")
  ngram_index_sequence = tf.gather(ngram_index, tf.slice(indices_to_gather, [0, 0], [-1, length]),
                                   check_indices, name="ngram_index_sequence")
  ngram_sequence = tf.gather(ngram_db, ngram_index_sequence, check_indices)
  # [batch_size] : int32
  # NOTE: batch_max_range is used instead of remaining_batch_lengths to prevent passing of the final EOS as the input
  consumed_sequence_lengths = tf.minimum(max_range, length, name="consumed_sequence_lengths")
  # [batch_size] : bool
  finished_batch_mask = tf.greater_equal(consumed_sequence_lengths, max_range, name="finished_batch_mask")
  input_sequence = tf.slice(sequence, [0, 0], [-1, length])
  expected_sequence = tf.slice(sequence, [0, 1], [-1, length])
  return input_sequence, expected_sequence, ngram_index_sequence, ngram_sequence, consumed_sequence_lengths, finished_batch_mask

def make_state_tuple(zero_state, variable):
  if variable:
    return tuple(LSTMStateTuple(*[tf.Variable(zero_state, trainable=False, name="network_state_{}_{}".format(part, layer))
                     for part in "ch"]) for layer in range(layer_count))
  else:
    return tuple(LSTMStateTuple(*[zero_state for part in "ch"]) for layer in range(layer_count))

class Train:
  with tf.name_scope('Train'):

    # [batch_size, network_size] : float32
    initial_zero_state = tf.zeros([batch_size, network_size], name="initial_zero_state")

    # tuple2(tuple2([batch_size, network_size])) : float32
    network_state = make_state_tuple(initial_zero_state, True)

    global_step = tf.Variable(1, dtype=tf.int32, name='global_step')

    # [batch_size] : int32
    random_batch_indices = tf.random_uniform([batch_size], minval=0, maxval=train_test_divider, dtype=tf.int32,
                                             name="random_batch_indices")
    # [batch_size] : int32
    batch_indices = tf.Variable(random_batch_indices, trainable=False, name="batch_indices")

    # [batch_size] : int32
    zero_batch = tf.zeros([batch_size], tf.int32, name="zero_batch")

    # [batch_size] : int32
    batch_progress = tf.Variable(zero_batch, trainable=False, name="batch_progress")

    input_sequence, expected_sequence, ngram_index_sequence, ngram_sequence, consumed_sequence_lengths, finished_batch_mask =\
      build_batch(batch_indices, unrolled_iterations, progress=batch_progress)

    # [batch_size, unrolled_iterations, symbol_count]
    input_vector = build_input_vector(input_sequence, ngram_index_sequence, ngram_sequence)

    with tf.variable_scope("Cell", reuse=None):
      cell = make_cell(dropout=True)
      outputs, state = tf.nn.dynamic_rnn(cell, input_vector,
                                         sequence_length=consumed_sequence_lengths,
                                         initial_state=network_state)

    total_loss, average_loss = get_loss(input_sequence, ngram_sequence, outputs, expected_sequence, consumed_sequence_lengths)

    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    tvars = tf.trainable_variables()
    gradients_with_vars = optimizer.compute_gradients(total_loss, var_list=tvars)
    #global_norm = tf.global_norm([gv[0] for gv in gradients_with_vars])
    apply_gradients = optimizer.apply_gradients(gradients_with_vars, global_step=global_step)

    # [batch_size] : int32
    final_batch_progress = tf.add(batch_progress, consumed_sequence_lengths, name="final_batch_progress")

    with tf.control_dependencies([apply_gradients]):
      copy_network_state_ops = []
      for layer_var, layer_new in zip(network_state, state):
        for state_part_var, state_part_new in zip(layer_var, layer_new):
          op = tf.assign(state_part_var, tf.select(finished_batch_mask, initial_zero_state, state_part_var))
          copy_network_state_ops.append(op)
      copy_network_state_op = tf.group(*copy_network_state_ops, name="copy_network_state_op")

      advance_batch_indices_op = tf.assign(batch_indices, tf.select(finished_batch_mask, random_batch_indices, batch_indices), name="advance_batch_indices_op")
      advance_batch_progress_op = tf.assign(batch_progress, tf.select(finished_batch_mask, zero_batch, final_batch_progress), name="advance_batch_progress_op")
      advance_batch_op = tf.group(advance_batch_indices_op, advance_batch_progress_op, name="advance_batch_op")

    train_op = tf.group(apply_gradients, advance_batch_op, copy_network_state_op)

    summary_list = []
    summary_list.append(tf.scalar_summary('train_average_loss', average_loss))
    #train_summary_list.append(tf.scalar_summary('global_norm', global_norm))
    summaries = tf.merge_summary(summary_list)

class Test:
  with tf.name_scope("Test"):
    indices = tf.range(train_test_divider, sequence_count)
    max_length = int(np.max(lengths[train_test_divider:]) + 2)
    input_sequence, expected_sequence, ngram_index_sequence, ngram_sequence, consumed_sequence_lengths, finished_mask =\
      build_batch(indices, max_length)
    input_vector = build_input_vector(input_sequence, ngram_index_sequence, ngram_sequence)
    initial_state = make_state_tuple(tf.zeros([test_count, network_size]), False)
    with tf.variable_scope("Cell", reuse=True):
      cell = make_cell(dropout=False)
      outputs, _ = tf.nn.dynamic_rnn(cell, input_vector,
                                     sequence_length=consumed_sequence_lengths,
                                     initial_state=initial_state)
    total_loss, average_loss = get_loss(input_sequence, ngram_sequence, outputs, expected_sequence, consumed_sequence_lengths)
    summaries = tf.merge_summary([tf.scalar_summary('test_average_loss', average_loss)])

init_op = tf.initialize_all_variables()


request = []

def tf_job(*args):
  def _tf_job(job):
    job.request = args
    return job
  return _tf_job

def tf_run_jobs(session, *jobs):
  request = []
  slices = []
  for job in jobs:
    a = len(request)
    if hasattr(job, 'request'):
      request.extend(job.request)
    b = len(request)
    slices.append((a,b,job))
  result = session.run(request)
  for a, b, job in slices:
    job(*result[a:b])

test_summary_job = lambda: None
train_summary_job = lambda: None
flush_summaries = lambda: None
if args.run_name:
  summary_dir = "summaries/" + args.run_name
  if os.path.exists(summary_dir):
    summary_writer = tf.train.SummaryWriter(summary_dir)
  else:
    summary_writer = tf.train.SummaryWriter(summary_dir, tf.get_default_graph())
  @tf_job(Train.summaries, Train.global_step)
  def train_summary_job(summary, step):
    summary_writer.add_summary(summary, step)
  @tf_job(Test.summaries, Train.global_step)
  def test_summary_job(summary, step):
    summary_writer.add_summary(summary, step)
  def flush_summaries():
    summary_writer.flush()

@tf_job(Train.train_op)
def train_job(_):
  pass

@tf_job(Test.average_loss)
def test_job(average_loss):
  print()
  print("Test average loss:", average_loss)

with tf.Session() as session:
  with status('Initializing...'):
    session.run(init_op)

  print("Trainable variables:")
  for v in tf.trainable_variables():
    print(v.name, v.get_shape())

  with status('Training...'):
    for i in range(2001):
      log("Training... {}/{}".format(i, 2000))
      tf_run_jobs(session, train_job, train_summary_job)
      if i % 20 == 0:
        tf_run_jobs(session, test_job, test_summary_job)
      flush_summaries()