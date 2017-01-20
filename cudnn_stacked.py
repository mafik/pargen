#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function

from itertools import chain
from libxml2mod import last

import data, datetime, argparse, sys

import numpy as np
import tensorflow as tf

from utils import *
from kneser_ney import KneserNeyLM
from nltk.util import ngrams as ngram_generator
from collections import Counter

parser = argparse.ArgumentParser(description='Neural language model')
parser.add_argument("-n", "--run-name", type=str, default=None)
parser.add_argument("-a", "--learning-rate", type=float, default=0.01)
parser.add_argument("-l", "--layer-count", type=int, default=2)
parser.add_argument("-s", "--network-size", type=int, default=200)
parser.add_argument("-d", "--dropout", type=float, default=0.5)
args = parser.parse_args()

if args.run_name == 'date':
  args.run_name = datetime.datetime.now().strftime("%A %H%M")

with status("Reading NKJP..."):
  arrays, lengths, symbol_map, reverse_symbol_map, symbol_count = data.read_nkjp_simple()

dataset_np = np.concatenate([np.concatenate(([data.GO], arr, [data.EOS])) for arr in arrays])
train_test_divider = int(len(lengths) * 0.80)
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

check_indices = False
network_size = args.network_size
layer_count = args.layer_count
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


input_vector_size = symbol_count

config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.3

with tf.device('/gpu:0'):
  cudnn_cell = tf.contrib.cudnn_rnn.CudnnLSTM(layer_count, network_size, input_vector_size, dropout=0)
  if args.dropout:
    print('A')
    cudnn_train_cell = tf.contrib.cudnn_rnn.CudnnLSTM(layer_count, network_size, input_vector_size, dropout=args.dropout)
  else:
    print('B')
    cudnn_train_cell = cudnn_cell

  config.graph_options.optimizer_options.opt_level = -1
  with tf.Session(config=config) as session:
    cudnn_param_size = cudnn_cell.params_size().eval()
  config.graph_options.optimizer_options.opt_level = 0
  cudnn_params = tf.Variable(tf.random_normal([cudnn_param_size], mean=0.0, stddev=0.005), validate_shape=False, name="cudnn_params")

params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
  cudnn_cell.params_to_canonical,
  cudnn_cell.canonical_to_params,
  cudnn_params)
tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)

output_projection_w = tf.Variable(tf.random_normal([network_size, symbol_count]), name="output_projection_w")
output_projection_b = tf.Variable(tf.random_normal([symbol_count]), name="output_projection_b")

saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

def build_input_vector(input_sequence, ngram_input_sequence, ngram_predictions):
  return tf.one_hot(input_sequence, symbol_count)


def get_total_loss(input_sequence, ngram_predictions, outputs, expected_sequence):
  outputs = tf.add(outputs, tf.log(ngram_predictions))
  # [batch_size, unrolled_iterations]
  losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=expected_sequence)
  losses = tf.where(tf.equal(input_sequence, data.EOS), tf.zeros_like(losses), losses)
  return tf.reduce_sum(losses)

def get_average_loss(total_loss, consumed_sequence_lengths):
  return total_loss / tf.cast(tf.reduce_sum(consumed_sequence_lengths), dtype=tf.float32)

def build_batch(indices, length, progress=None):
  if progress == None:
    progress = tf.zeros_like(indices)
  lengths = tf.gather(sequence_lengths, indices, check_indices, name="lengths")
  start_offsets = tf.gather(sequence_offsets, indices, check_indices, name="start_offsets")
  current_offsets = tf.add(start_offsets, progress, name="current_offsets")
  remaining_lengths = tf.sub(lengths, progress, name="remaining_lengths")
  max_range = tf.sub(remaining_lengths, 1)

  range_matrix = tf.minimum(*tf.meshgrid(tf.range(length + 1), max_range, indexing='ij'), name="range_matrix")
  indices_to_gather = tf.add(tf.tile(tf.expand_dims(current_offsets, 0), [length + 1, 1]),
                             range_matrix, name='indices_to_gather')
  sequence = tf.gather(dataset, indices_to_gather, check_indices, name="sequence")
  ngram_input_sequence = tf.gather(ngram_dataset, indices_to_gather[:-1],
                                   check_indices, name="ngram_input_sequence")
  ngram_predictions = tf.gather(ngram_probability_table, ngram_input_sequence[:,:,3], check_indices)
  # [batch_size] : int32
  # NOTE: batch_max_range is used instead of remaining_batch_lengths to prevent passing of the final EOS as the input
  consumed_sequence_lengths = tf.minimum(max_range, length, name="consumed_sequence_lengths")
  # [batch_size] : bool
  finished_batch_mask = tf.greater_equal(consumed_sequence_lengths, max_range, name="finished_batch_mask")
  input_sequence = sequence[:-1]
  expected_sequence = sequence[1:]
  return input_sequence, expected_sequence, ngram_input_sequence, ngram_predictions, consumed_sequence_lengths, finished_batch_mask

class Train:
  with tf.name_scope('Train'):
    batch_size = 200
    unrolled_iterations = 600
    initial_state = tf.zeros([layer_count * 1, batch_size, network_size])
    network_state_h = tf.Variable(initial_state, trainable=False, name="network_state_h")
    network_state_c = tf.Variable(initial_state, trainable=False, name="network_state_c")

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

    # [unrolled_iterations, batch_size, symbol_count]
    input_vector = build_input_vector(input_sequence, ngram_input_sequence, ngram_predictions)

    with tf.variable_scope("Cell", reuse=None):
      outputs, output_h, output_c = cudnn_train_cell(input_vector, network_state_h, network_state_c, cudnn_params)

    tiled_projection_matrix = tf.tile(tf.expand_dims(output_projection_w, 0), [unrolled_iterations, 1, 1])
    outputs = tf.matmul(outputs, tiled_projection_matrix) + output_projection_b

    total_loss = get_total_loss(input_sequence, ngram_predictions, outputs, expected_sequence)
    average_loss = get_average_loss(total_loss, consumed_sequence_lengths)

    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    tvars = tf.trainable_variables()
    cudnn_params.get_shape().as_list = lambda: [cudnn_param_size]

    #gradients = tf.gradients([total_loss], [cudnn_params])
    #apply_gradients = optimizer.apply_gradients([(gradients[0], cudnn_params)], global_step=global_step)

    gradients_with_vars = optimizer.compute_gradients(total_loss, var_list=tvars)
    apply_gradients = optimizer.apply_gradients(gradients_with_vars, global_step=global_step)

    # [batch_size] : int32
    final_batch_progress = tf.add(batch_progress, consumed_sequence_lengths, name="final_batch_progress")

    with tf.control_dependencies([apply_gradients, final_batch_progress]):
      copy_network_state_ops = []
      expanded_finished_batch_mask = tf.tile(tf.reshape(finished_batch_mask, [1, batch_size, 1]), [layer_count, 1, network_size])
      for state, out in zip([network_state_c, network_state_h], [output_c, output_h]):
        op = tf.assign(state, tf.where(expanded_finished_batch_mask, initial_state, out))
        copy_network_state_ops.append(op)
      copy_network_state_op = tf.group(*copy_network_state_ops, name="copy_network_state_op")

      advance_batch_indices_op = tf.assign(batch_indices, tf.where(finished_batch_mask, random_batch_indices, batch_indices), name="advance_batch_indices_op")
      advance_batch_progress_op = tf.assign(batch_progress, tf.where(finished_batch_mask, zero_batch, final_batch_progress), name="advance_batch_progress_op")
      advance_batch_op = tf.group(advance_batch_indices_op, advance_batch_progress_op, name="advance_batch_op")

    train_op = tf.group(apply_gradients, advance_batch_op, copy_network_state_op)

    summary_list = []
    summary_list.append(tf.summary.scalar('average_loss', average_loss))
    #train_summary_list.append(tf.summary.scalar('global_norm', global_norm))
    summaries = tf.summary.merge(summary_list)

class Test:
  with tf.name_scope("Test"):
    LOOPS = 6
    losses = []
    for i in range(LOOPS):
      a = train_test_divider + test_count * i / LOOPS
      b = train_test_divider + test_count * (i+1) / LOOPS
      indices = tf.range(a, b)
      max_length = int(np.max(lengths[a:b])) + 2
      input_sequence, expected_sequence, ngram_input_sequence, ngram_predictions, consumed_sequence_lengths, finished_mask =\
        build_batch(indices, max_length)
      input_vector = build_input_vector(input_sequence, ngram_input_sequence, ngram_predictions)
      initial_state = tf.zeros([layer_count * 1, b-a, network_size])
      with tf.variable_scope("Cell", reuse=True):
        outputs, output_h, output_c = cudnn_cell(input_vector, initial_state, initial_state, cudnn_params, is_training=False)
      tiled_projection_matrix = tf.tile(tf.expand_dims(output_projection_w, 0), [max_length, 1, 1])
      outputs = tf.matmul(outputs, tiled_projection_matrix) + output_projection_b
      average_loss = get_average_loss(get_total_loss(input_sequence, ngram_predictions, outputs, expected_sequence), consumed_sequence_lengths)
      losses.append(average_loss)

    summaries = tf.summary.merge([tf.summary.scalar('average_loss', tf.add_n(losses) / LOOPS)])

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
    job(session, *result[a:b])

test_summary_job = lambda: None
train_summary_job = lambda: None
flush_summaries = lambda: None
best_test_loss = 10000
last_test_loss = 0
train_loss = 0
if args.run_name:
  summary_dir = "summaries/" + args.run_name
  if os.path.exists(summary_dir):
    summary_writer = tf.summary.FileWriter(summary_dir)
  else:
    summary_writer = tf.summary.FileWriter(summary_dir, tf.get_default_graph())
  @tf_job(Train.summaries, Train.global_step)
  def train_summary_job(session, summary, step):
    summary_writer.add_summary(summary, step)
  @tf_job(Test.summaries, Train.global_step)
  def test_summary_job(session, summary, step):
    summary_writer.add_summary(summary, step)
  def flush_summaries():
    summary_writer.flush()

@tf_job(Train.train_op, Train.average_loss)
def train_job(session, _, average_loss):
  global train_loss
  train_loss = average_loss

@tf_job(Test.average_loss)
def test_job(session, average_loss):
  global best_test_loss, last_test_loss
  last_test_loss = average_loss
  if average_loss < best_test_loss:
    best_test_loss = average_loss
    if args.run_name:
      saver.save(session, os.path.join(summary_dir, "best_params"))
      open(os.path.join(summary_dir, "best_params_loss.txt"), 'w').write(str(average_loss) + '\n')
  #print("\nTest average loss:", average_loss)


if __name__ == "__main__":
  with tf.Session(config=config) as session:

    with status('Initializing...'):
      session.run(tf.global_variables_initializer())
    #print("Trainable variables:")
    #for v in tf.trainable_variables():
    #  print(v.name, v.get_shape())

    steps = 6000
    test_every_n_steps = 50
    with status('Training...'):
      for i in range(steps):
        log("Training... {}/{}, train loss: {:.3f}, test loss: best {:.3f}, last {:.3f}".format(i + 1, steps, train_loss, best_test_loss, last_test_loss))
        tf_run_jobs(session, train_job, train_summary_job)
        if i % test_every_n_steps == test_every_n_steps - 1:
          tf_run_jobs(session, test_job, test_summary_job)
        flush_summaries()
