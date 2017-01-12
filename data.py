# -*- coding: utf-8 -*-
from __future__ import print_function

import itertools

from utils import *
from glob import glob
import random, xml.sax

def read_ptb():
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
  return train_set, test_set

EOS = 0
GO = 1
UNK = 2
_symbol_map_prefix = ['<EOS>', '<GO>', '<UNK>']

def read_nkjp_simple():
  cache_path = "data-nkjp/cache.npz"
  try:
    npzfile = np.load(cache_path)
    arrays = npzfile['arrays']
    lengths = npzfile['lengths']
    symbol_map = npzfile['symbol_map']
    symbol_count = npzfile['symbol_count']
  except IOError as e:
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
          self.current_text += content

    handler = NkjpHandler()
    parser.setContentHandler(handler)
    for filename in glob('data-nkjp/*/text.xml'):
      parser.parse(filename)
    rand = random.Random(1337)
    random.shuffle(texts, random=lambda: rand.uniform(0,1))

    from collections import Counter, defaultdict
    symbol_counter = Counter()
    for text in texts:
      symbol_counter.update(text)
    symbol_map = np.array(_symbol_map_prefix + [symbol for symbol, count in symbol_counter.items() if count >= 5])
    def constant_factory(value):
      return itertools.repeat(value).next
    reverse_symbol_map = defaultdict(constant_factory(UNK))
    for i, symbol in enumerate(symbol_map):
      reverse_symbol_map[symbol] = i

    lengths = np.array([len(text) for text in texts], dtype=np.int32)
    arrays = np.array([np.array([reverse_symbol_map[c] for c in text], dtype=np.int32) for text in texts])
    symbol_count = len(symbol_map)
    np.savez(cache_path, arrays=arrays, lengths=lengths, symbol_map=symbol_map, symbol_count=symbol_count)
  return arrays, lengths, symbol_map, symbol_count

def read_nkjp(ngram_length=1):
  cache_path = "data-nkjp/cache-{}-gram.npz".format(ngram_length)
  texts = []
  try:
    npzfile = np.load(cache_path)
    arrays = npzfile['arrays']
    lengths = npzfile['lengths']
    symbol_map = npzfile['symbol_map']
    symbol_count = npzfile['symbol_count']
  except IOError as e:
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

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
          self.current_text += content

    handler = NkjpHandler()
    parser.setContentHandler(handler)
    for filename in glob('data-nkjp/*/text.xml'):
      parser.parse(filename)
    rand = random.Random(1337)
    random.shuffle(texts, random=lambda: rand.uniform(0,1))

    from collections import Counter, defaultdict
    symbol_counter = Counter()
    for text in texts:
      symbol_counter.update(text)
    symbol_map = np.array(_symbol_map_prefix + [symbol for symbol, count in symbol_counter.items() if count >= 5])
    def constant_factory(value):
      return itertools.repeat(value).next
    reverse_symbol_map = defaultdict(constant_factory(2))
    for i, symbol in enumerate(symbol_map):
      reverse_symbol_map[symbol] = i

    lengths = np.array([len(text)+2 for text in texts], dtype=np.int32)

    symbol_sequences = [[GO] * ngram_length + [reverse_symbol_map[c] for c in text] + [EOS] for text in texts]
    arrays = []
    def ngram_code(symbol_sequence, end_pos, length):
      if length == 1:
        return symbol_sequence[end_pos-1]
      else:
        ngram = tuple(symbol_sequence[end_pos - length:end_pos])
        if ngram not in reverse_symbol_map:
          i = len(reverse_symbol_map)
          reverse_symbol_map[ngram] = i
        return reverse_symbol_map[ngram]
    for symbol_sequence in symbol_sequences:
      data = [[ngram_code(symbol_sequence=symbol_sequence, end_pos=i, length=l) for l in range(1, ngram_length+1)]
              for i in range(ngram_length, len(symbol_sequence) + 1)]
      arrays.append(np.array(data, dtype=np.int32))
    arrays = np.array(arrays)
    symbol_count = len(reverse_symbol_map)
    np.savez(cache_path, arrays=arrays, lengths=lengths, symbol_map=symbol_map, symbol_count=symbol_count)

  train_cut = len(arrays) * 8 / 10
  valid_cut = len(arrays) * 9 / 10

  #for t in texts[valid_cut:]:
  #  print(t)
  train_set = Obj(arrays=arrays[:train_cut], lengths=lengths[:train_cut], size=train_cut)
  valid_set = Obj(arrays=arrays[train_cut:valid_cut], lengths=lengths[train_cut:valid_cut], size=valid_cut-train_cut)
  test_set = Obj(arrays=arrays[valid_cut:], lengths=lengths[valid_cut:], size=len(arrays) - valid_cut)
  return train_set, valid_set, test_set, symbol_map, symbol_count

if __name__ == "__main__":
  read_nkjp()
