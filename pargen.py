#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import xml.sax
import random
from itertools import chain

from time import clock
from glob import glob

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn
from tensorflow.python.ops import nn_ops

from math import *
from utils import *
import data

write("Reading training data... ")
train_set, valid_set, test_set, symbol_map = data.read_nkjp()
print("DONE")

# Training parameters

restore = True
train_time = float("inf") # 600
checkpoint_time = 600
test = True
gen = True
checkpoint_entropy = 999999

saver = None

# TODO: gen mode with bigger batch size

class Model:
  reuse_variables = None
  variables_loaded = False
  def __init__(self, mode):
    self.mode = mode
    write("Building tensorflow graph (mode = " + mode + ")... ")
    self.embedding_size = 50
    self.network_size = 200
    self.batch_size = 1 if self.mode == 'gen' else 50
    self.max_sequence_length = 1 if mode == 'gen' else 100
    dropout_keep_prob = 0.5 if mode == 'train' else 1 # 0.9
    self.num_layers = 2
    init_scale = 0.05
    initializer = tf.random_uniform_initializer(-init_scale, init_scale)

    self.sequence_lengths = tf.placeholder(tf.int32, [self.batch_size], "sequence_lengths")
    self.input_state = [tuple([tf.placeholder(tf.float32, [self.batch_size, self.network_size], "input_state") for part in "cm"]) for i in range(self.num_layers)]
    self.sequence = tf.placeholder(tf.int32, [self.batch_size, self.max_sequence_length], "sequence")
    self.chars = tf.placeholder(tf.int32, [], "chars")

    with tf.variable_scope("model", reuse=Model.reuse_variables, initializer=initializer):
      embedding_matrix = tf.get_variable("embedding", [len(symbol_map), self.embedding_size])
      cell = rnn_cell.LSTMCell(self.network_size, use_peepholes=True, cell_clip=10, state_is_tuple=True)
      if mode == 'train' and dropout_keep_prob < 1:
        cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
      if self.num_layers > 1:
        cell = rnn_cell.MultiRNNCell([cell] * self.num_layers, state_is_tuple=True)
      if self.network_size != self.embedding_size:
        cell = rnn_cell.InputProjectionWrapper(cell, num_proj=self.network_size)
      if len(symbol_map) != self.network_size:
        cell = rnn_cell.OutputProjectionWrapper(cell, output_size=len(symbol_map))
      self.cell = cell
      sequence_list = [tf.squeeze(t, [1]) for t in tf.split(1, self.max_sequence_length, self.sequence)]
      embeddings = [tf.nn.embedding_lookup(embedding_matrix, item) for item in (sequence_list[:-1] if self.mode != 'gen' else sequence_list)]
      self.logits, self.state = rnn.rnn(cell,
                                         embeddings,
                                         initial_state=self.input_state,
                                         sequence_length=self.sequence_lengths)
      #argmax = [tf.argmax(logit, 1) for logit in logits]
      if mode != 'gen':
        losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits[i], sequence_list[i + 1]) for i in range(self.max_sequence_length-1)]
        offset_costs = [tf.reduce_sum(loss) for loss in losses]
        self.total_cost = tf.add_n(offset_costs)
        self.cost_per_char = tf.div(self.total_cost, tf.cast(self.chars, tf.float32), "cost_per_char")
        tf.scalar_summary('cost_per_char', self.cost_per_char)
        tf.scalar_summary('total_cost', self.total_cost)
        tf.scalar_summary('chars', self.chars)
        self.summaries = tf.merge_all_summaries();
        if mode == 'train':
          self.lr = tf.Variable(0.0, trainable=False)
          optimizer = tf.train.AdagradOptimizer(self.lr)
          tvars = tf.trainable_variables()
          grad = tf.clip_by_global_norm(tf.gradients(self.total_cost, tvars), 5)[0]
          self.train_op = optimizer.apply_gradients(zip(grad, tvars))
        # TODO: add monitoring

    Model.reuse_variables = True # future invocations will reuse the same variables
    print("DONE")

  def zero_state(self):
    #return self.cell.zero_state(self.batch_size, np.float32)
    return [tuple([np.zeros([self.batch_size, self.network_size], dtype=np.float32) for part in "cm"]) for i in range(self.num_layers)]

  def train(self):
    global checkpoint_entropy
    start_clock = clock()
    last_checkpoint = 0
    i = 1
    initial_lr = 0.8
    falloff = 1
    train_writer = tf.train.SummaryWriter('summary', session.graph)

    t = clock() - start_clock
    while t < train_time:
      session.run(tf.assign(self.lr, initial_lr * (falloff ** (t / 10))))
      indicies = np.random.randint(len(train_set.lines), size=self.batch_size)
      args = {self.sequence_lengths: train_set.lengths[indicies]}

      for layer_placeholder, layer_state in zip(self.input_state, self.zero_state()):
        for part_placeholder, part_state in zip(layer_placeholder, layer_state):
          args[part_placeholder] = part_state

      args[self.sequence] = np.zeros((self.batch_size, self.max_sequence_length), dtype=np.int32)
      train_batch = train_set.arrays[indicies]
      chars = 0
      for sample_num in range(self.batch_size):
        x = min(self.max_sequence_length, len(train_batch[sample_num]))
        args[self.sequence][sample_num, :x] = train_batch[sample_num][:x]
        chars += x
      args[self.chars] = chars
      summaries, cost_per_char, train_op = session.run([self.summaries, self.cost_per_char, self.train_op], args)
      train_writer.add_summary(summaries, i)
      train_writer.flush()
      write("Network {}, training iteration {}, time {:.1f}, cost {:.3f}".format(loop + 1, i, t, cost_per_char))
      i += 1
      t = clock() - start_clock
      if t - last_checkpoint > checkpoint_time:
        validation_entropy = test_model.test(valid_set, False)

        write("Current per-character entropy = {:.3f}".format(validation_entropy))
        if validation_entropy < checkpoint_entropy:
          checkpoint_entropy = validation_entropy
          print(": saving new model")
          saver.save(session, "params/model.ckpt")
        else:
          print(": keeping old model with per-character entropy = {:.3f}".format(checkpoint_entropy))

        last_checkpoint = clock() - start_clock

  def test(self, dataset = test_set, verbose = True):
    offsets = np.zeros((self.batch_size,), dtype=np.int32)
    todo = [i for i in range(dataset.size)]
    test_results = 0.0
    chars = 0

    while len(todo):
      args = {
        self.sequence_lengths: np.zeros((self.batch_size,), dtype=np.int32),
        self.sequence: np.zeros((self.batch_size, self.max_sequence_length), dtype=np.int32)
      }

      for layer_placeholder, layer_state in zip(self.input_state, self.zero_state()):
        for part_placeholder, part_state in zip(layer_placeholder, layer_state):
          args[part_placeholder] = part_state

      n = min(self.batch_size, len(todo))
      for i in range(n):
        sample_num = todo[i]
        offset = offsets[i]
        x = min(self.max_sequence_length, dataset.lengths[sample_num] - offset)
        chars += x
        args[self.sequence_lengths][i] = x
        args[self.sequence][i, :x] = dataset.arrays[sample_num][offset:offset + x]
        offsets[i] += x
      for i in range(n, self.batch_size):
        args[self.sequence_lengths][i] = 0
      outputs = session.run([self.total_cost] + list(chain(*self.state)), args)
      cost, new_state = outputs[0], outputs[1:]
      for layer_placeholder in self.input_state:
        for part_placeholder in layer_placeholder:
          args[part_placeholder], new_state = new_state[0], new_state[1:]
      test_results += cost
      i = 0
      while i < n:
        sample_num = todo[i]
        if offsets[i] >= dataset.lengths[sample_num]:
          if len(todo) > n:
            todo[i] = todo.pop()
            offsets[i] = 0
            for layer_placeholder in self.input_state:
              for part_placeholder in layer_placeholder:
                args[part_placeholder][i, :] = 0#np.zeros((self.state_size,), dtype=np.float32)
            i += 1
          else:
            last = len(todo) - 1
            offsets[i] = offsets[last]
            todo[i] = todo[last]
            for layer_placeholder in self.input_state:
              for part_placeholder in layer_placeholder:
                args[part_placeholder][i, :] = args[part_placeholder][last,:]
                #args[self.input_state][i, :] = args[self.input_state][last, :]
            todo.pop()
            n -= 1
            # i stays the same
        else:
          i += 1
      if verbose:
        write("\rRemaining test sentences: {}/{}, per-character entropy: {:.3f}".format(len(todo), len(dataset.arrays), test_results / chars))
    if verbose:
      print()
    return test_results / chars

  def gen(self):
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
    #'''


g = tf.Graph()
with g.as_default():
  if train_time:
    train_model = Model('train')
  if test:
    test_model = Model('test')
  if gen:
    gen_model = Model('gen')

  with tf.Session() as session:
    test_result_list = []

    for loop in range(1):
      # print("Network {}...".format(loop + 1))
      saver = tf.train.Saver()
      with session.as_default():
        tf.initialize_all_variables().run(session=session)
        if restore:
          saver.restore(session, "params/model.ckpt")
          checkpoint_entropy = test_model.test(valid_set, False)
          print("Restored model with per-character entropy = {:.3f}".format(checkpoint_entropy))
        if train_time:
          train_model.train()
        if gen:
          gen_model.gen()
        if test:
          test_result_list.append(test_model.test())
    #if test:
    #  write("Entropy: {:.3f}, variance: {:.3f}\n".format(np.average(test_result_list), np.std(test_result_list)))
#print("DONE")

session.close()
