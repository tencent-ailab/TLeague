import pickle
import time
import numbers
import itertools
import uuid
import zmq
import numpy as np
from functools import partial, reduce
from tleague.utils import logger
from tleague.utils.tl_types import is_inherit
from tleague.utils.io import TensorZipper
from multiprocessing import Process
from collections import Counter
import multiprocessing
from threading import Thread
from tleague.utils import now


class ReplayMem(object):
  """Replay Memory. data_queue is a list of unrolls. Each unroll has unroll_length samples,
     Each batch samples several rollouts, where each rollout has rollout_length consecutive samples."""

  def __init__(self, rm_size, unroll_length, batch_size, rollout_length, log_infos_interval, learner_ports,
               unroll_num, info_stat, logger_dict, n_process=1, minimal_sample=None):
    self._rm_size = rm_size
    self._maxlen = int(np.ceil(rm_size / float(unroll_length)))
    if minimal_sample is not None:
      self._minimal_unroll = np.ceil(minimal_sample / float(unroll_length))
    else:
      self._minimal_unroll = self._maxlen
    self._unroll_length = unroll_length
    self._info_stat = info_stat
    self._batch_size = batch_size
    self._rollout_length = rollout_length
    assert unroll_length % rollout_length == 0
    assert batch_size % rollout_length == 0
    self._rollout_per_unroll = unroll_length // rollout_length
    self._data_queue = []
    self._next_idx = 0

    self._n_process = n_process
    self.logger_dict = logger_dict
    self._zmq_context = zmq.Context()
    self._log_infos_interval = log_infos_interval // n_process
    self._learner_ports = learner_ports

    self._pull_sockets = [self._zmq_context.socket(zmq.PULL) for _ in range(n_process)]
    self._output_eps = []
    for s in self._pull_sockets:
      s.setsockopt(zmq.RCVHWM, 1)
      s.bind("ipc:///tmp/" + str(uuid.uuid1())[:8])
      self._output_eps.append(s.getsockopt(zmq.LAST_ENDPOINT))
    print(self._output_eps)
    self._unroll_num = unroll_num
    self._ready = multiprocessing.RawValue('i', 0)  # do not sample before replay_memory is ready
    self._len = 0
    self._subps = []

  def __len__(self):
    return min(sum(data[0] for data in self._unroll_num), self._maxlen)

  def reset(self):
    self._stop_subprocess()
    for array in self._unroll_num:
      for i in range(len(array)):
        array[i] = 0
    self._ready = multiprocessing.RawValue('i', 0)
    self._start_subprocess()

  def ready_for_sample(self):
    return self._ready.value

  @staticmethod
  def _run_sub_rm(learner_port, output, maxlen, minlen, rollout_per_unroll,
                  rollout_length, batch_size, log_infos_interval, unroll_num, info_stat, logger_dict, ready_for_sample):
    p = RMFromQueue(learner_port, output, maxlen, minlen, rollout_per_unroll,
                    rollout_length, batch_size, log_infos_interval, unroll_num, info_stat, logger_dict, ready_for_sample)

  def _start_subprocess(self):
    max_len = int(np.ceil(self._maxlen / float(self._n_process)))
    min_len = int(np.ceil(self._minimal_unroll / float(self._n_process)))
    self._subps = [Process(target=self._run_sub_rm,
                           args=(self._learner_ports[i + 1], self._output_eps, max_len, min_len,
                                 self._rollout_per_unroll, self._rollout_length, self._batch_size,
                                 self._log_infos_interval, self._unroll_num[i], self._info_stat[i],
                                 self.logger_dict, self._ready)
                           ) for i in range(self._n_process)]
    for p in self._subps:
      p.start()

  def _stop_subprocess(self):
    for p in self._subps:
      p.terminate()

  def rollout_samplers(self):
    def _recv(s):
      msg = s.recv_multipart(copy=False)
      return tuple(pickle.loads(m) for m in msg)

    return [partial(_recv, s) for s in self._pull_sockets]


class RMFromQueue(object):
  """Replay Memory. data_queue is a list of unrolls. Each unroll has unroll_length samples,
     Each batch samples several rollouts, where each rollout has rollout_length consecutive samples."""

  def __init__(self, port, output_ep, maxlen, minlen, rollout_per_unroll, rollout_length, batch_size,
               log_infos_interval, shared_unroll_num, shared_info_stat, logger_dict,
               ready_for_sample, protect_unroll_num=10):
    self._maxlen = maxlen
    self._minlen = minlen
    self._rollout_length = rollout_length
    self._rollout_per_unroll = rollout_per_unroll
    self._rollout_per_batch = int(np.ceil(batch_size / float(rollout_length)))
    self._batch_size = batch_size
    self._protect_unroll_num = protect_unroll_num
    self._protect_unroll_num = max(protect_unroll_num, self._maxlen * 0.4 // 1)
    self._protect_rollout_num = min(protect_unroll_num, self._maxlen // 2) * self._rollout_per_unroll
    self._next_idx = 0
    self.idx = multiprocessing.RawValue('i', 0)
    self.start_train = multiprocessing.RawValue('i', 0)
    self._ready = ready_for_sample
    self._shared_unroll_num = shared_unroll_num
    self._info_stat = shared_info_stat
    self._unroll_num = 0
    self.logger_dict = logger_dict
    self.unroll_length = self._rollout_per_unroll * self._rollout_length
    self.shapes = tuple()
    self.data_queue = None
    self._log_infos_interval = log_infos_interval
    self.stat = Counter({})
    self._stat_key = list()
    self.stat_num = 0
    pull_data_thread = Thread(target=self._pull_data, args=(port,))
    pull_data_thread.start()

    while not self.shapes:
      time.sleep(0.2)
    self.dim = tuple(1 if not shape else reduce(lambda x, y: x * y, shape) for shape in self.shapes)
    self.data_queue = multiprocessing.RawArray('f', self._maxlen * self.unroll_length * sum(self.dim))
    push_data_process = multiprocessing.Process(target=self._sample_rollout, args=(output_ep,))
    push_data_process.start()

  def _pull_data(self, pull_port):
    zmq_context = zmq.Context()
    pull_socket = zmq_context.socket(zmq.PULL)
    pull_socket.setsockopt(zmq.RCVHWM, 1)
    pull_socket.bind("tcp://*:%s" % pull_port)
    sh_data_queue = None
    unroll_data_length = None

    # while True:
    #   if self.logger_dict["model_id"] is not None:
    #     model_id = self.logger_dict["model_id"]
    #     break
    #   time.sleep(1)

    while True:
      msg = pull_socket.recv(copy=False)
      try:
        data = pickle.loads(msg)
      except:
        logger.log(now() + 'In replay memory, ' +'data receive error, ' + 'unroll num: {}'.format(self._unroll_num))
        continue
      if data[0] == self.logger_dict["model_id"] or is_inherit(
          data[0], self.logger_dict["model_id"]):
        if sh_data_queue is None:
          if type(data[1]) is not np.ndarray:
            raise TypeError("TypeError: must be numpy ndarray, not {}".format(type(data[1])))
          self.shapes = data[3]
          # data[0]: model_id; [1]: numpy array; [2]: info; [3]: shapes; [4]: batch_size
          if len(data) == 5:
            self.data_batch_size = data[4]
          else:
            self.data_batch_size = 1
          while self.data_queue is None:
            time.sleep(0.2)
          sh_data_queue = np.ctypeslib.as_array(self.data_queue)
          unroll_data_length = self.unroll_length * sum(self.dim)
        self._update_print_infos(data[2])
        self._unroll_num += self.data_batch_size
        self._shared_unroll_num[0] = self._unroll_num
        if (self._next_idx + self.data_batch_size) <= self._maxlen:
            sh_data_queue[self._next_idx * unroll_data_length: (self._next_idx
                + self.data_batch_size) * unroll_data_length] = data[1]
        else:
            remainder = self._next_idx + self.data_batch_size - self._maxlen
            sh_data_queue[self._next_idx * unroll_data_length:] = \
                    data[1][0: (self.data_batch_size - remainder) * unroll_data_length]
            sh_data_queue[0: remainder * unroll_data_length] = \
                    data[1][(self.data_batch_size - remainder) * unroll_data_length:]
        if (self._next_idx + self.data_batch_size) >= self._maxlen:
            self._ready.value = True
            self.start_train.value = True
        self._next_idx = self._unroll_num % self._maxlen
        self.idx.value = self._next_idx
      else:
        self._shared_unroll_num[1] += self.data_batch_size 
        logger.log('Receive data from model ' + data[0] +
             ', while current model is ' + self.logger_dict["model_id"])

  def _sample_rollout(self, output_eps):
    zmq_context = zmq.Context()
    push_socket = zmq_context.socket(zmq.PUSH)
    push_socket.setsockopt(zmq.SNDHWM, 1)
    for output_ep in output_eps:
      push_socket.connect(str(output_ep, encoding="utf-8"))
    sh_data_queue = np.ctypeslib.as_array(self.data_queue)
    rollout_per_buffer = self._maxlen * self._rollout_per_unroll
    p = np.array([i >= self._protect_rollout_num for i in range(rollout_per_buffer)]) / (
        rollout_per_buffer - self._protect_rollout_num)
    replay_memory = sh_data_queue.reshape(rollout_per_buffer, self._rollout_length * sum(self.dim))
    necessary_protect_num = 0
    while not self.start_train.value:
      time.sleep(1)

    while True:
      last_writing_idx = self.idx.value
      batch_data = replay_memory[
        np.random.choice(rollout_per_buffer, self._rollout_per_batch, p=np.roll(p, last_writing_idx))].copy()
      current_writing_idx = self.idx.value
      read_unroll_num = current_writing_idx - last_writing_idx if current_writing_idx >= last_writing_idx \
          else self._maxlen + current_writing_idx - last_writing_idx
      necessary_protect_num = max(necessary_protect_num, read_unroll_num)
      if (necessary_protect_num > self._protect_unroll_num):
          logger.log(
              "The number of required protected unroll is {} while the number of protected unroll is set to {}".format(
                  necessary_protect_num, self._protect_unroll_num))
      push_socket.send_multipart(
        [pickle.dumps(d.reshape(self._batch_size, *shape)) for d, shape in
         zip(np.hsplit(batch_data.reshape(-1, sum(self.dim)), list(itertools.accumulate(self.dim))),
             self.shapes)], copy=False)

  @staticmethod
  def _decode_sample(sample):
    return TensorZipper.decompress(sample)

  @staticmethod
  def _filter(d):
    # filter the zstat infos, which can not average
    return dict([(_k, _v) for _k, _v in d.items()
                 if isinstance(_v, numbers.Number)])

  def _update_print_infos(self, infos):
    for info in infos:
      self.stat.update(Counter(self._filter(info)))
    self.stat_num += float(len(infos))
    if self.stat_num > self._log_infos_interval:
      for k, v in self.stat.items():
        self.stat[k] = float(int(v / self.stat_num * 100)) / 100.0  # two significant digits
      if not self._stat_key:
        self._stat_key = sorted(list(self.stat.keys()))
        self.logger_dict["stat_key"] = self._stat_key
      for i, key in enumerate(self._stat_key):
        self._info_stat[i] = self.stat[key]
      self.stat = Counter({})
      self.stat_num = 0
