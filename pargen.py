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

def write(s):
  sys.stdout.write("\r" + ' ' * 80)
  sys.stdout.write("\r" + s)
  sys.stdout.flush()

def weighted_choice(weights):
    totals = np.cumsum(weights)
    norm = totals[-1]
    throw = np.random.rand() * norm
    return np.searchsorted(totals, throw)

class Obj:
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      self.__dict__[k] = v

write("Reading training data... ")

if False:
  alphabet = ' abcdefghijklmnopqrstuwvqxyz0123456789\'.-$&#\\/*NX'
  symbol_map = {i+2: char for i, char in enumerate(alphabet)}
  symbol_map[0], symbol_map[1] = '<EOS>', '<GO>'
  reverse_symbol_map = {v:k for k,v in symbol_map.items()}
  def read_dataset(name):
    lines = [line.strip().replace("<unk>", "X") for line in open('data-ptb/' + name + '.txt').readlines()]
    lines = list(sorted(lines, key=len, reverse=True))
    arrays = []
    lengths = np.zeros((len(lines),), dtype=np.int32)
    for i, line in enumerate(lines):
      data = [1] + list(map(lambda x: reverse_symbol_map[x], line)) + [0]
      lengths[i] = len(data)
      arrays.append(np.array(data, dtype=np.int32))
    arrays = np.array(arrays)
    return Obj(lines=lines, arrays=arrays, lengths=lengths, size=len(lines))
  train_set = read_dataset('train')
  test_set = read_dataset('test')
else:
  alphabet = u' abcdefghijklmnopqrstuwvqxyząćęłóńśźżABCDEFGHIJKLMNOPQRSTUWVQXYZĄĆĘŁÓŃŚŹŻ0123456789\'",.?!~+-=_^$%&#@\\/*:;()[]{}«»<>„“”§|\n²ŢţÑÜäëöüčěřšČçâêôûàòèáÉéúŕýí°'
  symbol_map = {i+2: char for i, char in enumerate(alphabet)}
  symbol_map[0], symbol_map[1] = '<EOS>', '<GO>'
  reverse_symbol_map = {v:k for k,v in symbol_map.items()}
  parser = xml.sax.make_parser()
  parser.setFeature(xml.sax.handler.feature_namespaces, 0)

  texts = []

  class NkjpHandler( xml.sax.ContentHandler ):
    def __init__(self):
      self.current_text = None
      self.append = False
    def startElement(self, tag, attributes):
      if tag == 'div':
        self.current_text = ''
      elif tag == 'ab':
        self.append = True
    def endElement(self, tag):
      if tag == 'div':
        texts.append(self.current_text)
      elif tag == 'ab':
        self.append = False
    def characters(self, content):
      if self.append:
        if self.current_text:
          self.current_text += '\n'
        self.current_text += content.\
          replace(u'\xa0', u' ').\
          replace(u'–', '-').\
          replace(u'—', '-').\
          replace(u'\u02dd', '"').\
          replace(u'’', '\'').\
          replace(u'‘', '\'').\
          replace(u'…', '...').\
          replace(u'¨', '').\
          replace(u'•', '-').\
          replace(u'­', '-').\
          replace(u'`', '\'').\
          replace(u'−', '-').\
          replace(u'·', '-').\
          replace(u'´', '\'').\
          replace(u'×', 'x').\
          replace(u'ō', u'ö').\
          replace(u'ő', u'ö')

  handler = NkjpHandler()
  parser.setContentHandler(handler)
  for filename in glob('data-nkjp/*/text.xml'):
    parser.parse(filename)
  rand = random.Random(1337)
  random.shuffle(texts, random=lambda: rand.uniform(0,1))
  arrays = []
  lengths = np.zeros((len(texts),), dtype=np.int32)
  for i, text in enumerate(texts):
    try:
      data = [1] + list(map(lambda x: reverse_symbol_map[x], text)) + [0]
    except KeyError as e:
      print(u"Unknown symbol: '{}'".format(e.args[0]))
    lengths[i] = len(data)
    arrays.append(np.array(data, dtype=np.int32))
  arrays = np.array(arrays)
  cut = len(arrays) * 9 / 10
  train_set = Obj(lines=texts[:cut], arrays=arrays[:cut], lengths=lengths[:cut], size=cut)
  test_set = Obj(lines=texts[cut:], arrays=arrays[cut:], lengths=lengths[cut:], size=len(texts) - cut)

print("DONE")

restore = True
train_time = 0#float('inf')

checkpoint_time = 300

for network_size in [200]:
  print('network_size', network_size)
  for batch_size in [100]:
    print('batch_size', batch_size)
    for max_sequence_length in [100]:
      print('max_sequence_length', max_sequence_length)
      # TODO: benchmark dropout_keep_prob in [1.0, 0.9, 0.8, 0.5, 0.3]
      for dropout_keep_prob in [0.9]:
        print('dropout_keep_prob', dropout_keep_prob)

        g = tf.Graph()
        with g.as_default():

          write("Building tensorflow graph... ")
          num_layers = 2
          init_scale = 0.05
          initializer = tf.random_uniform_initializer(-init_scale, init_scale)

          state_size = network_size * num_layers * 2
          sequence_lengths = tf.placeholder(tf.int32, [batch_size], "sequence_lengths")
          input_state = tf.placeholder(tf.float32, [batch_size, state_size], "input_state")
          sequence = tf.placeholder(tf.int32, [batch_size, max_sequence_length], "sequence")

          embedding_matrix = tf.get_variable("embedding", [len(symbol_map), network_size])
          softmax_w = tf.get_variable("softmax_w", [network_size, len(symbol_map)])
          softmax_b = tf.get_variable("softmax_b", [len(symbol_map)])
          # TODO: replace softmax_w & b with output projection

          for training, reuse in [(True, None), (False, True)]:
            if train_time == 0:
              reuse = None
              if training:
                continue
            with tf.variable_scope("model", reuse=reuse, initializer=initializer):
              cell = rnn_cell.LSTMCell(network_size, use_peepholes=True, cell_clip=10)
              if training and dropout_keep_prob > 0:
                cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
              if num_layers > 1:
                cell = rnn_cell.MultiRNNCell([cell] * num_layers)
              #state_size = cell.zero_state(batch_size, tf.float32).get_shape()[1]
              sequence_list = [tf.squeeze(t, [1]) for t in tf.split(1, max_sequence_length, sequence)]
              embeddings = [tf.nn.embedding_lookup(embedding_matrix, item) for item in sequence_list[:-1]]
              outputs, state = rnn.rnn(cell,
                                       embeddings,
                                       initial_state=input_state,
                                       sequence_length=sequence_lengths)
              logits = [tf.nn.xw_plus_b(output, softmax_w, softmax_b) for output in outputs]
              argmax = [tf.argmax(logit, 1) for logit in logits]
              losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits[i], sequence_list[i + 1]) for i in range(max_sequence_length-1)]
              offset_costs = [tf.reduce_sum(loss) for loss in losses]
              total_cost = tf.add_n(offset_costs)
              if training:
                lr = tf.Variable(0.0, trainable=False)
                optimizer = tf.train.AdagradOptimizer(lr)
                tvars = tf.trainable_variables()
                grad = tf.clip_by_global_norm(tf.gradients(total_cost, tvars), 5)[0]
                train_op = optimizer.apply_gradients(zip(grad, tvars))
              #for offset_cost in offset_costs:
              #  tf.scalar_summary("offset_cost", offset_cost)
              #tf.scalar_summary("total_cost", total_cost)
              #summaries = tf.merge_all_summaries()
              #summary_writer = tf.train.SummaryWriter("logs")
              #summary_writer.add_graph(g)
              # TODO: add monitoring
              model = Obj(total_cost=total_cost, state=state)
              if training:
                train_model = model
              else:
                test_model = model

          print("DONE")

          with tf.Session() as session:

            for falloff in [1.0]: # 0.95
              print('falloff', falloff)
              for initial_lr in [0.8]:
                print('initial_lr', initial_lr)
                test_result_list = []

                for loop in range(1):
                  write("Network {}...".format(loop + 1))

                  with session.as_default():
                    tf.initialize_all_variables().run(session=session)
                    saver = tf.train.Saver()

                    # TRAINING
                    if restore:
                      saver.restore(session, "params/model.ckpt")
                    start_clock = clock()
                    last_checkpoint = 0
                    i = 1
                    chars = 0

                    t = clock() - start_clock
                    while t < train_time:
                      session.run(tf.assign(lr, initial_lr * (falloff ** (t / 10))))
                      indicies = np.random.randint(len(train_set.lines), size=batch_size)   
                      args = { sequence_lengths: train_set.lengths[indicies] } 
                      args[input_state] = np.zeros([batch_size, state_size], dtype=np.float32)
                      args[sequence] = np.zeros((batch_size, max_sequence_length), dtype=np.int32)
                      train_batch = train_set.arrays[indicies]
                      chars = 0
                      for sample_num in range(batch_size):
                        x = min(max_sequence_length, len(train_batch[sample_num]))
                        args[sequence][sample_num,:x] = train_batch[sample_num][:x]
                        chars += x
                      results = session.run([train_model.total_cost, train_op], args)
                      write("Network {}, training iteration {}, time {:.1f}, cost {:.3f}".format(
                          loop + 1, i, t, results[0] / chars))
                      i += 1
                      t = clock() - start_clock
                      if t - last_checkpoint > checkpoint_time:
                        save_path = saver.save(session, "params/model.ckpt")
                        last_checkpoint = t

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

                    ''' # 4.580
                    offsets = np.zeros((test_set.size,), dtype=np.int32)
                    states = np.zeros((test_set.size, state_size), dtype=np.float32)
                    todo = [i for i in range(test_set.size)]
                    test_results = 0.0
                    chars = 0

                    while len(todo):
                      args = {
                        sequence_lengths: np.zeros((batch_size,), dtype=np.int32),
                        input_state: np.zeros((batch_size, state_size), dtype=np.float32),
                        sequence: np.zeros((batch_size, max_sequence_length), dtype=np.int32)
                      }
                      n = min(batch_size, len(todo))
                      for i in range(n):
                        sample_num = todo[i]
                        offset = offsets[sample_num]
                        x = min(max_sequence_length, test_set.lengths[sample_num] - offset)
                        chars += x
                        args[sequence_lengths][i] = x
                        args[input_state][i,:] = states[sample_num,:]
                        args[sequence][i,:x] = test_set.arrays[sample_num][offset:offset + x]
                        offsets[sample_num] += x
                      out_state, cost = session.run([test_model.state, test_model.total_cost], args)
                      test_results += cost
                      for i in range(n):
                        sample_num = todo[i]
                        states[sample_num,:] = out_state[i,:]
                      todo = [i for i in todo if offsets[i] < test_set.lengths[i]]
                      write("\rTesting network {}... {}/{}: {:.3f}".format(loop + 1, len(todo), len(test_set.arrays), test_results / chars))
                    '''

                    # TESTING

                    offsets = np.zeros((batch_size,), dtype=np.int32)
                    todo = [i for i in range(test_set.size)]
                    test_results = 0.0
                    chars = 0

                    while len(todo):
                      args = {
                        sequence_lengths: np.zeros((batch_size,), dtype=np.int32),
                        input_state: np.zeros((batch_size, state_size), dtype=np.float32),
                        sequence: np.zeros((batch_size, max_sequence_length), dtype=np.int32)
                      }
                      n = min(batch_size, len(todo))
                      for i in range(n):
                        sample_num = todo[i]
                        offset = offsets[i]
                        x = min(max_sequence_length, test_set.lengths[sample_num] - offset)
                        chars += x
                        args[sequence_lengths][i] = x
                        args[sequence][i,:x] = test_set.arrays[sample_num][offset:offset + x]
                        offsets[i] += x
                      for i in range(n, batch_size):
                        args[sequence_lengths][i] = 0
                      args[input_state], cost = session.run([test_model.state, test_model.total_cost], args)
                      test_results += cost
                      i = 0
                      while i < n:
                        sample_num = todo[i]
                        if offsets[i] >= test_set.lengths[sample_num]:
                          if len(todo) > n:
                            todo[i] = todo.pop()
                            offsets[i] = 0
                            args[input_state][i,:] = np.zeros((state_size,), dtype=np.float32)
                            i += 1
                          else:
                            last = len(todo) - 1
                            offsets[i] = offsets[last]
                            todo[i] = todo[last]
                            args[input_state][i,:] = args[input_state][last,:]
                            todo.pop()
                            n -= 1
                            # i stays the same
                        else:
                          i += 1
                      write("\rTesting network {}... {}/{}: {:.3f}".format(loop + 1, len(todo), len(test_set.arrays), test_results / chars))
                      
                    test_result_list.append(test_results / chars)
                if test_result_list:
                  write("Entropy: {:.3f}, variance: {:.3f}\n".format(np.average(test_result_list), np.std(test_result_list)))
print("DONE")

session.close()
