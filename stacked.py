#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function

import data, datetime, argparse

import numpy as np
import tensorflow as tf

from tensorflow.python.ops.rnn_cell import *
from utils import *
from kneser_ney import KneserNeyLM
from nltk.util import ngrams

parser = argparse.ArgumentParser(description='Neural language model')
parser.add_argument("RUN_NAME", type=str, default = datetime.datetime.now().strftime("%A %H%M"))
parser.add_argument("-a", "--learning-rate", type=float, default=0.01)
args = parser.parse_args()

with status("Reading NKJP..."):
  arrays, lengths, symbol_map, symbol_count = data.read_nkjp_simple()

'''
with status("Loading n-gram model..."):
  n = 5
  kn_cache_path = "kn_cache_{}.npz".format(n)
  try:
    npzfile = np.load(kn_cache_path)
    lm = []
    for i in range(n):
      order = dict()
      for record in npzfile['arr_' + str(i)]:
        order[tuple(map(int, tuple(record)[:-1]))] = float(record[-1])
      lm.append(order)

  except IOError as e:
    log("Building n-gram model from scratch...")
    nkjp_ngrams = (ngram for array in arrays for ngram in ngrams(array, n, True, True, data.GO, data.EOS))
    lm = KneserNeyLM(n, nkjp_ngrams, data.GO, data.EOS).lm
    saved_arrays = []
    for i, order in enumerate(lm):
      dtype = ",".join(["u1"] * (n - i)) + ",f4"
      arr = np.array([key + (value,) for key, value in order.items()], dtype=dtype)
      saved_arrays.append(arr)
    np.savez(kn_cache_path, *saved_arrays)

def logprob(ngram):
  for i, order in enumerate(lm):
    if ngram[i:] in order:
      return order[ngram[i:]]
'''

batch_size = 25
sequence_count = len(lengths)
check_indices = True
unrolled_iterations = 5
network_size = 200
layer_count = 2
embedding_size = network_size

concatenated_arrays = np.concatenate([np.concatenate(([data.GO], arr, [data.EOS])) for arr in arrays])

# TODO: sprawdzic int8
# [sum(len(a) + 2 for a in arrays)] : int32
dataset = tf.Variable(concatenated_arrays,
                      trainable=False, dtype=tf.int32, name="flat_dataset")

# [num_sequences] : int32
sequence_lengths = tf.Variable([l+2 for l in lengths], trainable=False, name="sequence_lengths")

# [num_sequences] : int32
sequence_offsets = tf.cumsum(sequence_lengths, exclusive=True, name="sequence_offsets")

embedding_matrix = tf.Variable(tf.random_normal([symbol_count, embedding_size]), name='embedding_matrix')

with tf.name_scope('training'):
  global_step = tf.Variable(1, dtype=tf.int32, name='global_step')

  # [batch_size] : int32
  random_batch_indices = tf.random_uniform([batch_size], minval=0, maxval=sequence_count, dtype=tf.int32,
                                           name="random_batch_indices")
  # [batch_size] : int32
  batch_indices = tf.Variable(random_batch_indices, trainable=False, name="batch_indices")

  # [batch_size] : int32
  zero_batch = tf.zeros([batch_size], tf.int32, name="zero_batch")

  # [batch_size] : int32
  batch_progress = tf.Variable(zero_batch, trainable=False, name="batch_progress")

  # [batch_size] : int32
  batch_lengths = tf.gather(sequence_lengths, batch_indices, check_indices, name="batch_lengths")

  # [batch_size] : int32
  batch_start_offsets = tf.gather(sequence_offsets, batch_indices, check_indices, name="batch_start_offsets")

  # [batch_size] : int32
  current_batch_offsets = tf.add(batch_start_offsets, batch_progress, name="current_batch_offsets")

  # [batch_size] : int32
  remaining_batch_lengths = tf.sub(batch_lengths, batch_progress, name="remaining_batch_lengths")

  # [batch_size] : int32
  batch_max_range = tf.sub(remaining_batch_lengths, 1)

  # [batch_size, unrolled_iterations + 1] : int32
  batch_range_matrix = tf.minimum(*tf.meshgrid(batch_max_range, tf.range(unrolled_iterations + 1), indexing='ij'),
                                  name="batch_range_matrix")

  # [batch_size, unrolled_iterations + 1] : int32
  indices_to_gather = tf.add(
      tf.tile(tf.expand_dims(current_batch_offsets, 1), [1, unrolled_iterations + 1]),
      batch_range_matrix, name='indices_to_gather')

  # [batch_size, unrolled_iterations + 1] : int32
  sequence = tf.gather(dataset, indices_to_gather, check_indices, name="sequence")

  # [batch_size] : int32
  consumed_sequence_lengths = tf.minimum(remaining_batch_lengths, unrolled_iterations)

  # [batch_size, network_size] : float32
  initial_zero_state = tf.zeros([batch_size, network_size], name="initial_zero_state")

  # tuple2(tuple2([batch_size, network_size])) : float32
  network_state = tuple(LSTMStateTuple(*[tf.Variable(initial_zero_state, trainable=False, name="network_state_{}_{}".format(part, layer)) for part in "ch"]) for layer in range(layer_count))

cell = LSTMCell(network_size, use_peepholes=True, cell_clip=10, state_is_tuple=True)
cell = DropoutWrapper(cell, output_keep_prob=.5)
cell = MultiRNNCell([cell] * layer_count, state_is_tuple=True)
#cell = InputProjectionWrapper(cell, num_proj=network_size)
cell = OutputProjectionWrapper(cell, output_size=symbol_count)

input_sequence = tf.slice(sequence, [0, 0], [batch_size, unrolled_iterations])

# [batch_size, unrolled_iterations, embedding_size]
embeddings = tf.nn.embedding_lookup(embedding_matrix, input_sequence)

outputs, state = tf.nn.dynamic_rnn(cell, embeddings,
                                   sequence_length=consumed_sequence_lengths,
                                   initial_state=network_state,
                                   parallel_iterations=batch_size)

expected_sequence = tf.slice(sequence, [0, 1], [batch_size, unrolled_iterations])

# [batch_size, unrolled_iterations]
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(outputs, expected_sequence)

total_loss = tf.reduce_sum(losses)
average_loss = total_loss / tf.cast(tf.reduce_sum(consumed_sequence_lengths), dtype=tf.float32)

optimizer = tf.train.AdamOptimizer(args.learning_rate)
tvars = tf.trainable_variables()
gradients_with_vars = optimizer.compute_gradients(total_loss, var_list=tvars)
global_norm = tf.global_norm([gv[0] for gv in gradients_with_vars])
apply_gradients = optimizer.apply_gradients(gradients_with_vars, global_step=global_step)

init_op = tf.initialize_all_variables()
train_op = tf.group(apply_gradients)

summary_list = []

summary_list.append(tf.scalar_summary('average_loss', average_loss))
summary_list.append(tf.scalar_summary('global_norm', global_norm))

summaries = tf.merge_summary(summary_list)

summary_writer = tf.train.SummaryWriter("summaries/" + args.RUN_NAME,
                                        tf.get_default_graph())

with tf.Session() as session:
  session.run(init_op)
  '''
  print('Sequence offsets:')
  print(session.run(sequence_offsets))
  print('Batch progress:')
  print(session.run(batch_progress))
  print('Batch start offsets:')
  print(session.run(batch_start_offsets))
  print("Batch lengths:")
  print(session.run(batch_lengths))
  print("Current batch offsets:")
  print(session.run(current_batch_offsets))
  print("Batch max_range:")
  print(session.run(batch_max_range))
  print("Batch range matrix:")
  print(session.run(batch_range_matrix))
  print("Indices to gather:")
  print(session.run(indices_to_gather))
'''
  def dump():
    print("Trainable variables:")
    for v in tf.trainable_variables():
      print(v.name, v.get_shape())
    print("Sequence:")
    seq = session.run(sequence)
    for row in seq:
      print(row, ''.join(symbol_map[c] for c in row))
    print("Losses:")
    print(session.run(losses))
    print("Average loss:")
    print(session.run(average_loss))
  dump()
  print('####TRAINING#####')
  for i in range(200):
    summ, step, _ = session.run([summaries, global_step, train_op])
    summary_writer.add_summary(summ, step)
    summary_writer.flush()
  dump()
