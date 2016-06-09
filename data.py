from utils import *
from glob import glob
import random

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

def read_nkjp():
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
  return train_set, test_set, symbol_map