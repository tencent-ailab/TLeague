from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import zlib
import pickle

import numpy as np


class SparseTensor(object):

  def __init__(self, shape, index, value):
    self.shape = shape
    self.index = index
    self.value = value


class TensorZipper(object):

  @staticmethod
  def compress(data):
    return zlib.compress(pickle.dumps(data))

  @staticmethod
  def decompress(data):
    return pickle.loads(zlib.decompress(data))

  @staticmethod
  def compress_sparse(data):
    return zlib.compress(pickle.dumps(TensorZipper.dense_to_sparse(data)))

  @staticmethod
  def decompress_sparse(data):
    return TensorZipper.sparse_to_dense(pickle.loads(zlib.decompress(data)))

  @staticmethod
  def dense_to_sparse(data):
    if isinstance(data, np.ndarray) or isinstance(data, np.number):
      if len(data.shape) > 0:
        if all((dim < 65535 for dim in data.shape)):
          dtype = np.uint16
        else:
          dtype = np.uint32
        index = tuple(arr.astype(dtype) for arr in np.nonzero(data))
        value = data[index]
        return SparseTensor(data.shape, index, value)
      else:
        return data
    elif isinstance(data, tuple) or isinstance(data, list):
      return tuple(TensorZipper.dense_to_sparse(d) for d in data)
    elif isinstance(data, dict):
      return dict([(k, TensorZipper.dense_to_sparse(v)) for k, v in data.items()])
    else:
      raise TypeError("type %s not supported for compression" % type(data))

  @staticmethod
  def sparse_to_dense(data):
    if isinstance(data, SparseTensor):
      tensor = np.zeros(data.shape, dtype=data.value.dtype)
      tensor[data.index] = data.value
      return tensor
    elif isinstance(data, np.ndarray) or isinstance(data, np.number):
      return data
    elif isinstance(data, tuple) or isinstance(data, list):
      return tuple(TensorZipper.sparse_to_dense(d) for d in data)
    elif isinstance(data, dict):
      return dict([(k, TensorZipper.sparse_to_dense(v)) for k, v in data.items()])
    else:
      raise TypeError("type %s not supported for decompression" % type(data))
