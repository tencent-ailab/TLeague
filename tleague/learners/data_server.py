import pickle
import time
import numbers
from threading import Thread, Lock
from collections import deque
from collections import Counter

import tensorflow as tf
import zmq
from tleague.learners.replay_memory import ReplayMem, ImpSampReplayMem, ReplayMemMP
from tleague.utils import logger
from tleague.utils.io import TensorZipper
from tleague.utils.tl_types import is_inherit, ImitationTask
from tleague.utils.data_structure import ILData


class DataServer(object):
  """Data Server for Reinforcement Learning data providing.

  Prepare batch data, prefetch using tf.data.Dataset"""

  def __init__(self, learner_ports, rm_size, unroll_length, batch_size,
               ds, batch_worker_num=4, pull_worker_num=2,
               gpu_id_list=(0,), prefetch_buffer_size=None,
               rollout_length=1, version='v1', decode=False,
               log_infos_interval=20):
    print(ds.flatten_spec)
    shapes, dtypes = list(zip(*ds.flatten_spec))
    self.version = version
    if self.version == 'v2':
      shapes = tuple([(batch_size,) + tuple(s) for s in shapes])
    self._zmq_context = zmq.Context()
    self._pull_socket = self._zmq_context.socket(zmq.PULL)
    self._pull_socket.setsockopt(zmq.RCVHWM, 1)
    self._pull_socket.bind("tcp://*:%s" % learner_ports[1])
    self._pull_lock = Lock()
    self._model_id = None  # abandon the data from last task
    self._batch_size = batch_size
    self.dtypes = dtypes
    self.shapes = shapes
    self.unroll_num = 0
    self.aband_unroll_num = 0
    self._infos = deque(maxlen=max(100, log_infos_interval))
    self._log_infos_interval = log_infos_interval
    self.info_stat = {}
    self._info_num = 0
    self._batch_worker_num = batch_worker_num
    self._mk_rm(rm_size, unroll_length, batch_size, rollout_length,
                version, decode)

    self._pull_data_threads = []
    for i in range(pull_worker_num):
      self._pull_data_threads.append(Thread(target=self._pull_data,
                                            daemon=True))
    for thread in self._pull_data_threads:
      thread.start()

    gpu_num = len(gpu_id_list)
    num_dataset = max(gpu_num, 1)
    self.input_datas = []
    for i in range(num_dataset):
      if self.version == 'v2':
        dataset = tf.data.Dataset.range(batch_worker_num).apply(
          tf.contrib.data.parallel_interleave(
            lambda x: tf.data.Dataset.from_generator(
              self._data_generator, self.dtypes, self.shapes, args=(x,)),
            cycle_length=batch_worker_num,
            sloppy=True,
            buffer_output_elements=1))  # parallel generators
      else:
        dataset = tf.data.Dataset.range(batch_worker_num).apply(
          tf.contrib.data.parallel_interleave(
            lambda x: tf.data.Dataset.from_generator(
              self._data_generator, self.dtypes, self.shapes, args=(x,)).apply(
              tf.contrib.data.batch_and_drop_remainder(batch_size)),
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

  def _mk_rm(self, rm_size, unroll_length, batch_size,
             rollout_length, version, decode):
    if version == 'v2':
      self._replay_mem = ReplayMemMP(rm_size, unroll_length, batch_size,
                                     rollout_length, decode=decode,
                                     n_process=self._batch_worker_num)
    else:
      self._replay_mem = ReplayMem(rm_size, unroll_length, batch_size,
                                   rollout_length, decode=decode)

  def _update_model_id(self, model_id):
    self._model_id = model_id

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
    self.unroll_num = 0
    self.aband_unroll_num = 0

  def _data_getter(self):
    self._pull_lock.acquire()
    data = self._pull_socket.recv(copy=False)
    self._pull_lock.release()
    return pickle.loads(data)

  def _pull_data(self):
    while True:
      if self._model_id is None:
        time.sleep(1)
        continue
      data = self._data_getter()
      if data[0] == self._model_id or is_inherit(data[0], self._model_id):
        self._replay_mem.append(data[1])
        if self._log_infos_interval > 0:
          self._infos.extend(data[2])
          self._info_num += len(data[2])
          if self._info_num >= self._log_infos_interval:
            self._info_num -= self._log_infos_interval
            self._print_infos()
        self.unroll_num += 1
      else:
        self.aband_unroll_num += 1
        logger.log('Receive data from model ' + data[0] +
                   ', while current model is ' + self._model_id)

  def _print_infos(self):
    # filter the zstat infos, which can not average
    filter = lambda d: dict([(_k, _v) for _k, _v in d.items()
                             if isinstance(_v, numbers.Number)])
    stat = Counter({})
    for info in list(self._infos):
      stat.update(Counter(filter(info)))
    num = float(len(self._infos))
    for k, v in stat.items():
      stat[k] = float(int(v/num * 100)) / 100.0  # two significant digits
    self.info_stat = stat
    # logger.log(stat)

  def _data_generator(self, x):
    if self.version == 'v2':
      sampler = self._replay_mem.rollout_samplers()[x]
    else:
      sampler = self._replay_mem.sample_rollout
    while True:
      while not self._replay_mem.ready_for_sample():
        time.sleep(5)
      if self.version == 'v2':
        yield sampler()  # return one batch
      else:
        for sample in sampler():  # get one rollout
          yield sample  # return one sample


class ImDataServer(object):
  """Data Server for Imitation Learning data providing."""

  def __init__(self, ports, train_replay_filelist, val_replay_filelist,
               batch_size, min_train_sample_num, min_val_sample_num,
               ob_space, ac_space, rm_size=64000*32,
               unroll_length=32, train_generator_worker_num=4,
               val_generator_worker_num=2, pull_worker_num=2,
               repeat_training_task=False, use_gpu=True,
               rollout_length=1, lstm=False, hs_len=128,):
    self._train_replays = train_replay_filelist
    self._val_replays = val_replay_filelist
    self._replay_tasks = []
    for c in zip(self._val_replays, self._train_replays):
      self._replay_tasks.extend(c)
    self._replay_tasks.extend(self._train_replays[len(self._val_replays):])
    self._repeat_training_task = repeat_training_task

    self._zmq_context = zmq.Context()
    self._rep_socket = self._zmq_context.socket(zmq.REP)
    self._rep_socket.bind("tcp://*:%s" % ports[0])
    self._pull_socket = self._zmq_context.socket(zmq.PULL)
    self._pull_socket.setsockopt(zmq.RCVHWM, 1)
    self._pull_socket.bind("tcp://*:%s" % ports[1])
    self._message_thread = Thread(target=self._message_worker, daemon=True)
    self._pull_data_threads = [Thread(target=self._pull_data, daemon=True)
                               for _ in range(pull_worker_num)]
    self._train_rm = ImpSampReplayMem(rm_size, unroll_length, batch_size,
                                      rollout_length, min_train_sample_num)
    self._val_rm = ImpSampReplayMem(rm_size, unroll_length, batch_size,
                                    rollout_length, min_val_sample_num)
    self._num_train_samples = 0
    self._num_val_samples = 0

    self._pull_lock = Lock()
    self._message_thread.start()
    for thread in self._pull_data_threads:
      thread.start()
    self.ds = ILData(ob_space, ac_space, lstm, hs_len)
    shapes, dtypes = list(zip(*self.ds.flatten_spec))
    dtypes = (dtypes, tf.float32)  # sample weight
    shapes = (shapes, [])
    train_dataset = tf.data.Dataset.range(train_generator_worker_num).apply(
      tf.contrib.data.parallel_interleave(
        lambda x: tf.data.Dataset.from_generator(
          self._create_data_generator(self._train_rm),
          dtypes, shapes).apply(
          tf.contrib.data.batch_and_drop_remainder(batch_size)),
        cycle_length=train_generator_worker_num,
        sloppy=True,
        buffer_output_elements=1))
    # train_dataset = train_dataset.batch(batch_size)
    if use_gpu:
      prefetch_op = tf.contrib.data.prefetch_to_device(
        device="/gpu:0", buffer_size=train_generator_worker_num)
      train_dataset = train_dataset.apply(prefetch_op)
    else:
      train_dataset = train_dataset.prefetch(buffer_size=1)
    self._train_batch = train_dataset.make_one_shot_iterator().get_next()
    self.train_batch_input = self.ds.make_structure(self._train_batch[0])
    self.train_batch_weight = self._train_batch[1]

    val_dataset = tf.data.Dataset.range(val_generator_worker_num).apply(
      tf.contrib.data.parallel_interleave(
        lambda x: tf.data.Dataset.from_generator(
          self._create_data_generator(self._val_rm),
          dtypes, shapes).apply(
          tf.contrib.data.batch_and_drop_remainder(batch_size)),
        cycle_length=val_generator_worker_num,
        sloppy=True,
        buffer_output_elements=1))
    # val_dataset = val_dataset.batch(batch_size)
    self._val_batch = val_dataset.make_one_shot_iterator().get_next()
    self.val_batch_input = self.ds.make_structure(self._val_batch[0])
    self.val_batch_weight = self._val_batch[1]

  @property
  def ready_for_train(self):
    if not self._train_rm.ready_for_sample():
      logger.log(
        'train data queue not full ({}/{} unrolls, wait...)'.format(
          len(self._train_rm), self._train_rm._minimal_unroll)
      )
      return False
    else:
      return True

  @property
  def ready_for_val(self):
    if not self._val_rm.ready_for_sample():
      logger.log(
        'val data queue not full ({}/{} unrolls, wait...)'.format(
          len(self._val_rm), self._val_rm._minimal_unroll)
      )
      return False
    else:
      return True

  def _create_data_generator(self, rm):

    def data_generator():
      while True:
        while not rm.ready_for_sample():
          time.sleep(5)
        for sample, weight in rm.sample_rollout():
          yield (TensorZipper.decompress(sample), weight)

    return data_generator

  def _pull_data(self):
    while True:
      self._pull_lock.acquire()
      msg = self._pull_socket.recv(copy=False)
      self._pull_lock.release()
      replay_task, data, weights = pickle.loads(msg)
      if not isinstance(replay_task, ImitationTask):
        raise RuntimeError(
          "received task %s not recognized".format(replay_task))
      if not replay_task.validation:
        self._train_rm.append(data, weights)
        self._num_train_samples += len(data)
      else:
        self._val_rm.append(data, weights)
        self._num_val_samples += len(data)

  def _message_worker(self):
    train_idx = 0
    l = len(self._train_replays)
    while True:
      msg = self._rep_socket.recv_string()
      if msg == 'replay_task':
        if len(self._replay_tasks) > 0:
          self._rep_socket.send_pyobj(self._replay_tasks.pop(0))
        elif self._repeat_training_task:
          if train_idx == 0:
            logger.log('Training task empty, repeat training task.')
          self._rep_socket.send_pyobj(self._train_replays[train_idx])
          train_idx = (train_idx + 1) % l
        else:
          self._rep_socket.send_pyobj("")
      else:
        raise RuntimeError("message not recognized")
