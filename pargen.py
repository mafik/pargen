#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import random
import math

import model
import utils
import gen

default_paraphrase = u"Niestety, psy czasem się mylą. Może ich zgubić choćby zapach palącego papierosy."

parser = argparse.ArgumentParser(description='Paraphrase generator')
utils.add_arguments(parser)
model.add_arguments(parser)
parser.add_argument("--paraphrase", type=unicode, default=default_paraphrase)
parser.add_argument("--favorize-alpha", type=float, default=2.0)

args = parser.parse_args()
scale = 20

def semantic_distance(x, y):
  a = set(x.split())
  b = set(y.split())
  common = a & b
  if len(common) == 0:
    return 0.0
  c = float(len(common))
  S = 0.5 * math.log(len(a) / c, 2) + 0.5 * math.log(len(b) / c, 2)
  if S < 1.0:
    return S * scale
  return math.exp(-3 * S) * scale

start_sentence = args.paraphrase
sentence = start_sentence
old_similarity = 0.0 # semantic_distance(sentence, start_sentence)
print ("Start sentence:", start_sentence)
print ("Semantic distance", old_similarity)
generator = gen.SentenceGenerator(args)
generator.favorize(start_sentence.split(), args.favorize_alpha)

with utils.ModelSession(args) as model_session:
  session = model_session.session
  for i in range(100):
    T = 0.9 ** (i/ 100. * 10)
    removed_chars = int(math.ceil(len(sentence) * T))
    removal_start = random.randint(0, len(sentence) - removed_chars)
    removal_end = removal_start + removed_chars
    prefix = sentence[:removal_start]
    suffix = sentence[removal_end:]

    new_sentence = generator.run(session, prefix, suffix, removed_chars)
    new_similarity = semantic_distance(start_sentence, new_sentence)
    p = math.exp((new_similarity - old_similarity)/T)
    print(u"Result {}, generated with T={:.2} ({} chars), similarity={:.5}, p={:.2} : {}".format(
      i, T, removed_chars, new_similarity, p, new_sentence.replace('\n', '<NL>')))

    if new_similarity > old_similarity or random.random() < p:
      sentence = new_sentence
      old_similarity = new_similarity
      print("Swapping")
print ("Final result: " + sentence)
