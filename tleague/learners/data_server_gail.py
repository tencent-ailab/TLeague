import tensorflow as tf
import json
import os
import pandas as pd
import numpy as np
from tleague.utils import logger

class DataServerGAIL(object):
  """Data Server for Reinforcement Learning data providing.

  Prepare batch data, prefetch using tf.data.Dataset"""
  def __init__(self, batch_size,
               ds, batch_worker_num=4, gpu_id_list=(0,), prefetch_buffer_size=None,
               data_path=None):
    self._data_path = data_path
    assert data_path is not None
    shapes, dtypes = list(zip(*ds.flatten_spec))
    print(shapes, dtypes)
    shapes = tuple([(batch_size,) + tuple(s) for s in shapes])

    self._batch_size = batch_size
    self.dtypes = dtypes
    self.shapes = shapes
    self._batch_worker_num = batch_worker_num
    self.data = self._read_data()

    gpu_num = len(gpu_id_list)
    num_dataset = max(gpu_num, 1)
    self.input_datas = []
    for i in range(num_dataset):
      dataset = tf.data.Dataset.from_generator(
            self._data_generator, self.dtypes, self.shapes)
      # dataset = dataset.batch(self._batch_size)
      if gpu_num <= 0:
        dataset = dataset.prefetch(prefetch_buffer_size)
      else:
        gpu_id = gpu_id_list[i]
        prefetch_op = tf.contrib.data.prefetch_to_device(
          device="/gpu:" + str(gpu_id), buffer_size=prefetch_buffer_size)
        dataset = dataset.apply(prefetch_op)
      iterator = dataset.make_one_shot_iterator()
      self.input_datas.append(ds.make_structure(iterator.get_next()))

  def _read_data(self):
    if os.path.exists(self._data_path):
      if os.path.isdir(self._data_path):
        folder_name = [os.path.join(self._data_path, folder) for folder in os.listdir(self._data_path)]
        logger.log("Reading data from folders: {}".format(folder_name))
        file_name = [os.path.join(folder, file) for folder in folder_name
                     for file in os.listdir(folder) if file.endswith('.npy')]
      else:
        file_name = [self._data_path]
      data = np.vstack([np.load(f) for f in file_name])
      logger.log("expert data shape {}".format(data.shape))
    else:
      raise ValueError("invalid data path")
    return data

  def _data_generator(self):
    while True:
      batch_data = self.data[np.random.choice(self.data.shape[0], self._batch_size)]
      yield (batch_data, )
