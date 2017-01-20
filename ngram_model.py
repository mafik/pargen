import model

from utils import *

import data

with status("Loading n-gram model..."):
  n = 5
  ngram_db_path = "ngram_db_{}.npz".format(n)
  train_test_divider = int(len(data.lengths) * 0.80)
  try:
    npzfile = np.load(ngram_db_path)
    ngram_dataset = npzfile["ngram_dataset"]
    ngram_probability_table = npzfile["ngram_probability_table"]
  except IOError as e:
    from collections import Counter
    from nltk.util import ngrams as ngram_generator
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
