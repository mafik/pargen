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
parser.add_argument("-g", "--generate", type=str, default=None)
parser.add_argument("-a", "--learning-rate", type=float, default=0.01)
parser.add_argument("-i", "--bootstrap-in", type=bool, default=False)
parser.add_argument("-o", "--bootstrap-out", type=bool, default=False)
parser.add_argument("-m", "--bootstrap-mem", type=bool, default=False)
args = parser.parse_args()

if args.run_name == 'date':
  args.run_name = datetime.datetime.now().strftime("%A %H%M")

with status("Reading NKJP..."):
  arrays, lengths, symbol_map, symbol_count = data.read_nkjp_simple()

dataset_np = np.concatenate([np.concatenate(([data.GO], arr, [data.EOS])) for arr in arrays])
train_test_divider = int(len(lengths) * 0.8)
sequence_count = len(lengths)
test_count = sequence_count - train_test_divider

with status("Loading n-gram model..."):
  n = 5
  ngram_db_path = "ngram_db_{}.npz".format(n)
  try:
    npzfile = np.load(ngram_db_path)
    ngram_dataset = npzfile["ngram_dataset"]
    ngram_probability_table = npzfile["ngram_probability_table"]
  except IOError as e:
    log("Building n-gram model from scratch...")

    ngrams = tuple(Counter() for i in range(n+1))
    ngrams[n].update(ngram for array in arrays[:train_test_divider] for ngram in ngram_generator(chain((data.GO,)*n, array, (data.EOS,)), n))
    for i in range(n - 1, 0, -1):
      for ngram, count in ngrams[i+1].items():
        ngrams[i][ngram[1:]] += count

    log("Precomputing unique prefixes/suffixes")
    # unique_prefixes[i][ngram] where len(ngram) == i contains number of different symbols that proceed given ngram
    unique_prefixes = tuple(Counter() for i in range(n))
    # unique_suffixes[i][ngram] where len(ngram) == i contains number of different symbols that follow given ngram
    unique_suffixes = tuple(Counter() for i in range(n))
    for i in range(n, 0, -1):
      unique_prefixes[i-1].update(ngram[1:] for ngram in ngrams[i].keys())
      unique_suffixes[i-1].update(ngram[:-1] for ngram in ngrams[i].keys())

    log("Indexing ngrams")
    all_ngrams = tuple(set() for i in range(n+1))
    for array in arrays:
      for ngram in ngram_generator(chain((data.GO,)*n, array, (data.EOS,)), n):
        all_ngrams[n].add(ngram)
    for i in range(n - 1, 0, -1):
      for ngram in all_ngrams[i+1]:
        all_ngrams[i].add(ngram[1:])
    # maps ngram tuple to ngram number
    ngram_idx = tuple(dict() for i in range(n))
    for i in range(n, 0, -1):
      for num, ngram in enumerate(all_ngrams[i]):
        ngram_idx[i-1][ngram] = num

    discount = (1.0, 0.5, 0.75, 0.75, 0.75, 0.75)

    def prob(full_ngram):
      current_p = 0.0
      p_multiplier = 1.0
      estimation_base = ngrams
      for i in range(n, 0, -1):
        ngram = full_ngram[-i:]
        prefix = ngram[:-1]
        if estimation_base[i-1][prefix]:
          #print("i", i)
          #print("full ngram", full_ngram)
          #print("ngram", ngram)
          #print("prefix", prefix)
          #print("estamition_base", estimation_base[i][ngram])
          p = max(0, estimation_base[i][ngram] - discount[i]) / estimation_base[i-1][prefix]
          current_p += p * p_multiplier
          p_multiplier *= discount[i] / estimation_base[i-1][prefix] * unique_suffixes[i-1][prefix]
          estimation_base = unique_prefixes
      current_p += p_multiplier / symbol_count # probability of an unseen symbol
      #print(u"Prob of {}: {}".format(''.join(symbol_map[c] for c in ngram), prob_cache[ngram]))
      return current_p

    precomputing_started_clock = log("Precomputing successor probabilities")
    # probabilities for the next symbol based on the last (n-1)-gram
    ngram_probability_table = np.zeros((len(ngram_idx[-2]), symbol_count), dtype=np.float32)
    progress = 0
    for ngram, idx in ngram_idx[-2].items():
      if progress % 500 == 0:
        if progress:
          time_left = (clock() - precomputing_started_clock) / progress * (len(ngram_idx[-2]) - progress)
        else:
          time_left = float("inf")
        log("Precomputing successor probabilities: {:.1f}% ({:.0f} seconds left)".format(100.0 * progress / len(ngram_idx[-2]), time_left))
      probs = np.array([prob(ngram + (i,)) for i in range(symbol_count)])
      probs = probs / max(np.sum(probs), 0.0001)
      #pred_symbol = np.argmax(ngram_probability_table[ngram_idx[n1][prefix]])
      #print(u"Prefix '{}', prediction {:3d} '{}'".format(''.join(symbol_map[c] for c in prefix), pred_symbol, symbol_map[pred_symbol]))
      ngram_probability_table[idx, :] = probs
      progress += 1

    log("Building ngram sequence")
    ngram_dataset = np.zeros(dataset_np.shape + (n,), dtype=np.int32)
    offset = 0
    for array_index, arr in enumerate(arrays):
      for i in range(n):
        ngram_length = i + 1
        for pos, ngram in enumerate(ngram_generator(chain((data.GO,) * ngram_length, arr), ngram_length)):
          ngram_dataset[offset + pos, i] = ngram_idx[i][ngram]
      offset += len(arr) + 2

    np.savez(ngram_db_path, ngram_dataset=ngram_dataset, ngram_probability_table=ngram_probability_table)

def translate1(array):
  return ''.join(symbol_map[i] for i in array)

def translate2(array):
  return '\n'.join(translate1(row) for row in array)

batch_size = 100
check_indices = True
unrolled_iterations = 300
network_size = 500
layer_count = 2
ngram_count = len(ngram_probability_table)

# [sum(len(a) + 2 for a in arrays)] : int32
dataset = tf.Variable(dataset_np,
                      trainable=False, dtype=tf.int32, name="dataset")

ngram_dataset = tf.Variable(ngram_dataset, trainable=False, dtype=tf.int32, name="ngram_dataset")
ngram_probability_table = tf.Variable(ngram_probability_table, trainable=False, dtype=tf.float32, name="ngram_probability_table")

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

def build_input_vector(input_sequence, ngram_input_sequence, ngram_predictions):
  input_list = []
  input_list.append(tf.one_hot(input_sequence, symbol_count))
  if args.bootstrap_in:
    mean, variance = tf.nn.moments(ngram_predictions, [2], keep_dims=True)
    input_list.append(tf.nn.batch_normalization(ngram_predictions, mean, variance, 0, 1, 0.00001))
  if args.bootstrap_mem:
    input_list.append(tf.gather(ngram_embeddings, tf.squeeze(tf.slice(ngram_input_sequence, [0,0,3], [-1,-1,1]))))
  return input_list[0] if len(input_list) == 1 else tf.concat(2, input_list)


def get_loss(input_sequence, ngram_predictions, outputs, expected_sequence, consumed_sequence_lengths):
  if args.bootstrap_out:
    outputs = tf.add(outputs, tf.log(ngram_predictions))
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
  ngram_input_sequence = tf.gather(ngram_dataset, tf.slice(indices_to_gather, [0, 0], [-1, length]),
                                   check_indices, name="ngram_input_sequence")
  ngram_predictions = tf.gather(ngram_probability_table, tf.squeeze(tf.slice(ngram_input_sequence, [0, 0, 3], [-1, -1, 1]), [2]), check_indices)
  # [batch_size] : int32
  # NOTE: batch_max_range is used instead of remaining_batch_lengths to prevent passing of the final EOS as the input
  consumed_sequence_lengths = tf.minimum(max_range, length, name="consumed_sequence_lengths")
  # [batch_size] : bool
  finished_batch_mask = tf.greater_equal(consumed_sequence_lengths, max_range, name="finished_batch_mask")
  input_sequence = tf.slice(sequence, [0, 0], [-1, length])
  expected_sequence = tf.slice(sequence, [0, 1], [-1, length])
  return input_sequence, expected_sequence, ngram_input_sequence, ngram_predictions, consumed_sequence_lengths, finished_batch_mask

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

    global_step = tf.Variable(1, dtype=tf.int32, trainable=False, name='global_step')

    # [batch_size] : int32
    random_batch_indices = tf.random_uniform([batch_size], minval=0, maxval=train_test_divider, dtype=tf.int32,
                                             name="random_batch_indices")
    # [batch_size] : int32
    batch_indices = tf.Variable(random_batch_indices, trainable=False, name="batch_indices")

    # [batch_size] : int32
    zero_batch = tf.zeros([batch_size], tf.int32, name="zero_batch")

    # [batch_size] : int32
    batch_progress = tf.Variable(zero_batch, trainable=False, name="batch_progress")

    input_sequence, expected_sequence, ngram_input_sequence, ngram_predictions, consumed_sequence_lengths, finished_batch_mask =\
      build_batch(batch_indices, unrolled_iterations, progress=batch_progress)

    # [batch_size, unrolled_iterations, symbol_count]
    input_vector = build_input_vector(input_sequence, ngram_input_sequence, ngram_predictions)

    with tf.variable_scope("Cell", reuse=None):
      cell = make_cell(dropout=True)
      outputs, state = tf.nn.dynamic_rnn(cell, input_vector,
                                         sequence_length=consumed_sequence_lengths,
                                         initial_state=network_state)

    total_loss, average_loss = get_loss(input_sequence, ngram_predictions, outputs, expected_sequence, consumed_sequence_lengths)

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
    input_sequence, expected_sequence, ngram_input_sequence, ngram_predictions, consumed_sequence_lengths, finished_mask =\
      build_batch(indices, max_length)
    input_vector = build_input_vector(input_sequence, ngram_input_sequence, ngram_predictions)
    initial_state = make_state_tuple(tf.zeros([test_count, network_size]), False)
    with tf.variable_scope("Cell", reuse=True):
      cell = make_cell(dropout=False)
      outputs, _ = tf.nn.dynamic_rnn(cell, input_vector,
                                     sequence_length=consumed_sequence_lengths,
                                     initial_state=initial_state)
    total_loss, average_loss = get_loss(input_sequence, ngram_predictions, outputs, expected_sequence, consumed_sequence_lengths)
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

saver = tf.train.Saver()

with tf.Session() as session:
  with status('Initializing...'):
    session.run(init_op)

  if args.generate:
    saver.restore(session, "checkpoints/%s.ckpt" % (args.generate))


    write("Generating sentences...")
    args = {
      self.sequence_lengths: np.ones((self.batch_size,), dtype=np.int32),
      self.sequence: np.zeros((self.batch_size, self.max_sequence_length), dtype=np.int32)
    }
    results = []
    beam = []
    beam.append({
      "sentence": "",
      "entropy": 0,
      "state": self.zero_state(),
      "last_symbol": 1
    })
    beam_width = 100
    branching = 10
    iterations = 400 * beam_width
    for i in range(iterations):
      if i % 1000 == 0:
        write("Generating sentences... (iteration {:,}/{:,})".format(i, iterations))
      best = beam.pop()
      args[self.sequence][0, 0] = best["last_symbol"]
      for layer_placeholder, layer_state in zip(self.input_state, best["state"]):
        for part_placeholder, part_state in zip(layer_placeholder, layer_state):
          args[part_placeholder] = part_state
      outputs = session.run([self.logits[0]] + list(chain(*self.state)), args)
      generated_logits = outputs[0]
      generated_state = list(chunks(outputs[1:], 2))
      generated_distribution = np.exp(generated_logits[0, :])
      generated_distribution /= np.sum(generated_distribution)

      results.append({
        "sentence": best["sentence"],
        "entropy": (best["entropy"] - log(generated_distribution[0], 2)) / (len(best["sentence"]) + 1) # pow
      })
      results.sort(key=lambda x: -x["entropy"])
      results = results[-10:]

      for generated_arg in top_k(generated_distribution, branching):
        next = {
          "sentence": best["sentence"] + symbol_map[generated_arg],
          "entropy": best["entropy"] - log(generated_distribution[generated_arg], 2),
          "last_symbol": generated_arg,
          "state": generated_state
        }
        beam.append(next)
      if False:
        maxlen = min(map(lambda x: len(x["sentence"]), beam))
        beam.sort(key=lambda x: -x["entropy"] + (maxlen - len(x["sentence"])) * 0.1)
      elif False:
        beam.sort(key=lambda x: -x["entropy"] + (200 - len(x["sentence"])) * 0.1)
      else:
        beam.sort(key=lambda x: -x["entropy"])

      if len(beam) > beam_width:
        beam = beam[-beam_width:]
    write("Generating sentences...")
    print("DONE")
    '''
    print("Beam contents:")
    for result in beam[-10:]:
      print(u"{:.3f} : {}".format(result["entropy"] / (1+len(result['sentence'])), result["sentence"]))
    #'''
    print("Top complete sentences:")
    for result in results:
      print(u"{:.3f} : {}".format(result["entropy"], result["sentence"]))

  else:
    print("Trainable variables:")
    for v in tf.trainable_variables():
      print(v.name, v.get_shape())

    steps = 501
    with status('Training...'):
      for i in range(steps):
        log("Training... {}/{}".format(i + 1, steps))
        tf_run_jobs(session, train_job, train_summary_job)
        if (i + 1) % 50 == 0:
          save_path = saver.save(session, "checkpoints/%s-%04d.ckpt" % (args.run_name, i))
          tf_run_jobs(session, test_job, test_summary_job)
        flush_summaries()
