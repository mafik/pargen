#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import xml.sax
import random

from time import clock
from glob import glob

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn
from tensorflow.python.ops import nn_ops

from utils import *
import data

write("Reading training data... ")
train_set, test_set, symbol_map = data.read_nkjp()
print("DONE")

# Training parameters

restore = True
train_time = 300#float('inf')
checkpoint_time = 300


class Model:
  reuse_variables = None
  def __init__(self, mode):
    self.mode = mode
    write("Building tensorflow graph (mode = " + mode + ")... ")
    network_size = 200
    self.batch_size = 1 if self.mode == 'gen' else 100
    self.max_sequence_length = 1 if mode == 'gen' else 100
    dropout_keep_prob = 0.9 if mode == 'train' else 1
    num_layers = 2
    init_scale = 0.05
    initializer = tf.random_uniform_initializer(-init_scale, init_scale)

    self.state_size = network_size * num_layers * 2
    self.sequence_lengths = tf.placeholder(tf.int32, [self.batch_size], "sequence_lengths")
    self.input_state = tf.placeholder(tf.float32, [self.batch_size, self.state_size], "input_state")
    self.sequence = tf.placeholder(tf.int32, [self.batch_size, self.max_sequence_length], "sequence")

    embedding_matrix = tf.get_variable("embedding", [len(symbol_map), network_size])
    softmax_w = tf.get_variable("softmax_w", [network_size, len(symbol_map)])
    softmax_b = tf.get_variable("softmax_b", [len(symbol_map)])
    # TODO: replace softmax_w & b with output projection

    with tf.variable_scope("model", reuse=Model.reuse_variables, initializer=initializer):
      cell = rnn_cell.LSTMCell(network_size, use_peepholes=True, cell_clip=10)
      if mode == 'train' and dropout_keep_prob < 1:
        cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
      if num_layers > 1:
        cell = rnn_cell.MultiRNNCell([cell] * num_layers)
      sequence_list = [tf.squeeze(t, [1]) for t in tf.split(1, self.max_sequence_length, self.sequence)]
      embeddings = [tf.nn.embedding_lookup(embedding_matrix, item) for item in sequence_list[:-1]]
      outputs, self.state = rnn.rnn(cell,
                               embeddings,
                               initial_state=self.input_state,
                               sequence_length=self.sequence_lengths)
      logits = [tf.nn.xw_plus_b(output, softmax_w, softmax_b) for output in outputs]
      #argmax = [tf.argmax(logit, 1) for logit in logits]
      losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits[i], sequence_list[i + 1]) for i in range(self.max_sequence_length-1)]
      offset_costs = [tf.reduce_sum(loss) for loss in losses]
      self.total_cost = tf.add_n(offset_costs)
      if mode == 'train':
        self.lr = tf.Variable(0.0, trainable=False)
        optimizer = tf.train.AdagradOptimizer(self.lr)
        tvars = tf.trainable_variables()
        grad = tf.clip_by_global_norm(tf.gradients(self.total_cost, tvars), 5)[0]
        self.train_op = optimizer.apply_gradients(zip(grad, tvars))
      # TODO: add monitoring

    Model.reuse_variables = True # future invocations will reuse the same variables
    print("DONE")

  def train(self):
    start_clock = clock()
    last_checkpoint = 0
    i = 1
    initial_lr = 0.8
    falloff = 1

    t = clock() - start_clock
    while t < train_time:
      session.run(tf.assign(self.lr, initial_lr * (falloff ** (t / 10))))
      indicies = np.random.randint(len(train_set.lines), size=self.batch_size)
      args = {self.sequence_lengths: train_set.lengths[indicies]}
      args[self.input_state] = np.zeros([self.batch_size, self.state_size], dtype=np.float32)
      args[self.sequence] = np.zeros((self.batch_size, self.max_sequence_length), dtype=np.int32)
      train_batch = train_set.arrays[indicies]
      chars = 0
      for sample_num in range(self.batch_size):
        x = min(self.max_sequence_length, len(train_batch[sample_num]))
        args[self.sequence][sample_num, :x] = train_batch[sample_num][:x]
        chars += x
      results = session.run([self.total_cost, self.train_op], args)
      write("Network {}, training iteration {}, time {:.1f}, cost {:.3f}".format(
        loop + 1, i, t, results[0] / chars))
      i += 1
      t = clock() - start_clock
      if t - last_checkpoint > checkpoint_time:
        save_path = saver.save(session, "params/model.ckpt")
        last_checkpoint = t

  def test(self):
    offsets = np.zeros((self.batch_size,), dtype=np.int32)
    todo = [i for i in range(test_set.size)]
    test_results = 0.0
    chars = 0

    while len(todo):
      args = {
        self.sequence_lengths: np.zeros((self.batch_size,), dtype=np.int32),
        self.input_state: np.zeros((self.batch_size, self.state_size), dtype=np.float32),
        self.sequence: np.zeros((self.batch_size, self.max_sequence_length), dtype=np.int32)
      }
      n = min(self.batch_size, len(todo))
      for i in range(n):
        sample_num = todo[i]
        offset = offsets[i]
        x = min(self.max_sequence_length, test_set.lengths[sample_num] - offset)
        chars += x
        args[self.sequence_lengths][i] = x
        args[self.sequence][i, :x] = test_set.arrays[sample_num][offset:offset + x]
        offsets[i] += x
      for i in range(n, self.batch_size):
        args[self.sequence_lengths][i] = 0
      args[self.input_state], cost = session.run([self.state, self.total_cost], args)
      test_results += cost
      i = 0
      while i < n:
        sample_num = todo[i]
        if offsets[i] >= test_set.lengths[sample_num]:
          if len(todo) > n:
            todo[i] = todo.pop()
            offsets[i] = 0
            args[self.input_state][i, :] = np.zeros((self.state_size,), dtype=np.float32)
            i += 1
          else:
            last = len(todo) - 1
            offsets[i] = offsets[last]
            todo[i] = todo[last]
            args[self.input_state][i, :] = args[self.input_state][last, :]
            todo.pop()
            n -= 1
            # i stays the same
        else:
          i += 1
      write(
        "\rTesting network {}... {}/{}: {:.3f}".format(loop + 1, len(todo), len(test_set.arrays), test_results / chars))

    return test_results / chars


g = tf.Graph()
with g.as_default():
  train_model = Model('train')
  test_model = Model('test')
  gen_model = Model('gen')

  with tf.Session() as session:
    test_result_list = []

    for loop in range(1):
      write("Network {}...".format(loop + 1))

      with session.as_default():
        tf.initialize_all_variables().run(session=session)
        saver = tf.train.Saver()

        # TRAINING
        if restore:
          saver.restore(session, "params/model.ckpt")

        if train_time:
          train_model.train()

        # GENERATION
        '''
        print()

        print("Sample sentences:")
        for sentence_num in range(10):
          args = {
            sequence_lengths: np.ones((batch_size,), dtype=np.int32),
            input_state: np.zeros((batch_size, state_size), dtype=np.float32),
            sequence: np.zeros((batch_size, max_sequence_length), dtype=np.int32)
          }
          generated_sentence = []
          args[sequence][0,0] = 1
          for symbol_num in range(100):
            generated_logits, generated_state = session.run([logits[0], state], args)
            generated_distribution = np.exp(generated_logits[0,:])
            generated_distribution /= np.sum(generated_distribution)
            generated_arg = weighted_choice(generated_distribution)
            #generated_arg = np.argmax(generated_logits[0,:])
            if generated_arg == 0:
              break
            args[sequence][0,0] = generated_arg
            args[input_state] = generated_state
            generated_symbol = symbol_map[generated_arg]
            generated_sentence.append(generated_symbol)
          generated_sentence = ''.join(generated_sentence)

          print('"' + generated_sentence + '"')
        continue
        # '''

        # TESTING
        test_result_list.append(test_model.test())
    if test_result_list:
      write("Entropy: {:.3f}, variance: {:.3f}\n".format(np.average(test_result_list), np.std(test_result_list)))
print("DONE")

session.close()
