from __future__ import print_function
import numpy as np

def load(filename):
  npzfile = np.load(filename)
  index2label = npzfile['index2label']
  embeddings = npzfile['embeddings']
  label2index_map = {label:index for index, label in enumerate(index2label)}
  return label2index_map, index2label, embeddings

def load_update(filename, labels, size):
  npzfile = np.load(filename)
  index2label = np.array(labels, dtype=str)
  label2index_map = {label:index for index, label in enumerate(index2label)}
  embeddings = np.zeros((len(labels), size), dtype=np.float32)
  for i in range(npzfile['index2label'].shape[0]):
    label = npzfile['index2label'][i]
    if label not in label2index_map: continue
    index = label2index_map[label]
    embeddings[index,:] = npzfile['embeddings'][i,:]
  return label2index_map, index2label, embeddings

def save(filename, index2label, embeddings):
  assert len(index2label.shape) == 1
  assert len(embeddings.shape) == 2
  assert index2label.shape[0] == embeddings.shape[0]
  np.savez(filename, index2label=index2label, embeddings=embeddings)

if __name__ == "__main__":
  import time
  a = time.clock()
  n = 1000000
  dims = 100
  labels = np.array([str(i) for i in range(n)], dtype=str)
  saved = np.ones((n, dims), dtype=np.float32)
  save("test.npz", labels, saved)
  b = time.clock()
  load("test.npz")
  c = time.clock()
  print("Done!")
  print("Saving took", str(b-a), 'seconds')
  print("Loading took", str(c-b), 'seconds')
