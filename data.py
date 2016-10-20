# -*- coding: utf-8 -*-
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

def read_nkjp(ngram_length=1):

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

  symbol_map = {1: '<GO>', 0: '<EOS>'}
  reverse_symbol_map = {v:k for k,v in symbol_map.items()}

  def get_symbol_index(x):
    if x not in reverse_symbol_map:
      i = len(reverse_symbol_map)
      symbol_map[i] = x
      reverse_symbol_map[x] = i
    return reverse_symbol_map[x]

  def ngram_code(symbol_sequence, end_pos, length):
    if length == 1:
      return symbol_sequence[end_pos-1]
    else:
      ngram = tuple(symbol_sequence[end_pos - length:end_pos])
      return get_symbol_index(ngram)

  lengths = np.array([len(text)+2 for text in texts], dtype=np.int32)
  symbol_sequences = [[1] * ngram_length + list(map(get_symbol_index, text)) + [0] for text in texts]
  alphabetic_symbol_map = dict(symbol_map)
  arrays = []
  for symbol_sequence in symbol_sequences:
    data = [[ngram_code(symbol_sequence=symbol_sequence, end_pos=i, length=l) for l in range(1, ngram_length+1)]
            for i in range(ngram_length, len(symbol_sequence) + 1)]
    arrays.append(np.array(data, dtype=np.int32))
  arrays = np.array(arrays)
  train_cut = len(arrays) * 8 / 10
  valid_cut = len(arrays) * 9 / 10
  train_set = Obj(lines=texts[:train_cut], arrays=arrays[:train_cut], lengths=lengths[:train_cut], size=train_cut)
  valid_set = Obj(lines=texts[train_cut:valid_cut], arrays=arrays[train_cut:valid_cut], lengths=lengths[train_cut:valid_cut], size=valid_cut-train_cut)
  test_set = Obj(lines=texts[valid_cut:], arrays=arrays[valid_cut:], lengths=lengths[valid_cut:], size=len(texts) - valid_cut)
  return train_set, valid_set, test_set, alphabetic_symbol_map, len(symbol_map)