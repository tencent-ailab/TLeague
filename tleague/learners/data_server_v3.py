import time
import multiprocessing
from multiprocessing import Manager

import tensorflow as tf
from tleague.learners.replay_memory_v3 import ReplayMem
from tleague.utils import logger


class DataServer(object):
  """Data Server for Reinforcement Learning data providing.

  Prepare batch data, prefetch using tf.data.Dataset"""

  def __init__(self, learner_ports, rm_size, unroll_length, batch_size,
               ds, batch_worker_num=4, gpu_id_list=(0,), prefetch_buffer_size=None,
               rollout_length=1, log_infos_interval=20):
    print(ds.flatten_spec)
    shapes, dtypes = list(zip(*ds.flatten_spec))
    shapes = tuple([(batch_size,) + tuple(s) for s in shapes])
    self._model_id = None  # abandon the data from last task
    self._learner_ports = learner_ports

    manager = Manager()
    self.logger_dict = manager.dict({
      "model_id": None,
      "stat_key": set(),
    })
    self._stat_key = set()
    self._unroll_num = [multiprocessing.RawArray('i', 2) for _ in range(batch_worker_num)]
    self._info_stat = [multiprocessing.RawArray('f', 100) for _ in range(batch_worker_num)]

    self._batch_size = batch_size
    self.dtypes = dtypes
    self.shapes = shapes
    self._batch_worker_num = batch_worker_num
    self._mk_rm(rm_size, unroll_length, batch_size, rollout_length, log_infos_interval)

    gpu_num = len(gpu_id_list)
    num_dataset = max(gpu_num, 1)
    self.input_datas = []
    for i in range(num_dataset):
      dataset = tf.data.Dataset.range(batch_worker_num).apply(
        tf.contrib.data.parallel_interleave(
          lambda x: tf.data.Dataset.from_generator(
            self._data_generator, self.dtypes, self.shapes, args=(x,)),
          cycle_length=batch_worker_num,
          sloppy=True,
          buffer_output_elements=1))  # parallel generators
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

  def _mk_rm(self, rm_size, unroll_length, batch_size, rollout_length, log_infos_interval):
    self._replay_mem = ReplayMem(rm_size, unroll_length, batch_size,
                                 rollout_length, log_infos_interval, self._learner_ports,
                                 unroll_num=self._unroll_num, info_stat=self._info_stat, logger_dict=self.logger_dict,
                                 n_process=self._batch_worker_num)

  def _update_model_id(self, model_id):
    self.logger_dict["model_id"] = model_id

  @property
  def ready_for_train(self):
    if not self._replay_mem.ready_for_sample():
      logger.log(
        'train data queue not full ({}/{} unrolls, wait...)'.format(
          len(self._replay_mem), self._replay_mem._minimal_unroll)
      )
      return False
    else:
      return True

  def reset(self):
    """Clean the replay memory. Call before each learner task begins."""
    self._replay_mem.reset()
    for array in self._unroll_num:
      for i in range(len(array)):
        array[i] = 0

  def _data_generator(self, x):
    sampler = self._replay_mem.rollout_samplers()[x]
    while True:
      while not self._replay_mem.ready_for_sample():
        time.sleep(5)
      yield sampler()  # return one batch

  def __getattr__(self, item):
    if item == "unroll_num":
      return sum(data[0] for data in self._unroll_num)
    if item == "aband_unroll_num":
      return sum(data[1] for data in self._unroll_num)
    if item == "info_stat":
      info_stat = dict()
      if not self._stat_key:
        self._stat_key = self.logger_dict["stat_key"]
      for i, k in enumerate(self._stat_key):
        info_stat.update({k: sum(data[i] for data in self._info_stat) / len(self._info_stat)})
      return info_stat

  def __setattr__(self, key, value):
    logger_item = {'aband_unroll_num', 'unroll_num', 'info_stat'}
    if key in logger_item:
      msg = "can not set attributes to readonly attribute {}".format(key)
      raise AttributeError(msg)
    super().__setattr__(key, value)
