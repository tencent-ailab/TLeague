import pickle
import random
import time
import uuid
import zmq
import numpy as np
from threading import Thread, Lock
from tleague.utils.io import TensorZipper
from multiprocessing import Process

class ReplayMem(object):
  """Replay Memory. data_queue is a list of unrolls. Each unroll has unroll_length samples,
     Each batch samples several rollouts, where each rollout has rollout_length consecutive samples."""

  def __init__(self, rm_size, unroll_length, batch_size, rollout_length,
               decode=False, minimal_sample=None):
    self._rm_size = rm_size
    self._decode = decode  # whether store decoded samples in RM
    self._maxlen = int(np.ceil(rm_size / float(unroll_length)))
    if minimal_sample is not None:
      self._minimal_unroll = np.ceil(minimal_sample / float(unroll_length))
    else:
      self._minimal_unroll = self._maxlen
    self._unroll_length = unroll_length
    self._batch_size = batch_size
    self._rollout_length = rollout_length
    assert unroll_length % rollout_length == 0
    assert batch_size % rollout_length == 0
    self._rollout_per_unroll = unroll_length // rollout_length
    self._data_queue = []
    self._next_idx = 0
    self._ready = False  # do not sample before replay_memory is ready

  def __len__(self):
    return len(self._data_queue)

  def reset(self):
    self._ready = False
    time.sleep(1)
    self._data_queue = []
    self._next_idx = 0

  def ready_for_sample(self):
    return self._ready

  def append(self, data):  # append one unroll
    idx = self._next_idx
    self._next_idx = (self._next_idx + 1) % self._maxlen
    if self._decode:
      data = [self._decode_sample(d) for d in data]
    if self._maxlen > len(self._data_queue):
      self._data_queue.append(data)
      if not self._ready:
        self._ready = len(self._data_queue) >= self._minimal_unroll
    else:
      self._data_queue[idx] = data

  def rollout_samplers(self):
    return [self.sample_rollout]

  def sample_rollout(self):
    i = random.randint(0, len(self._data_queue) - 1)
    unroll = self._data_queue[i]  # backup the unroll in case of overwritten by new unroll
    j = random.randint(0, self._rollout_per_unroll - 1) * self._rollout_length
    if self._decode:
      for k in range(self._rollout_length):
        yield unroll[j + k]
    else:
      for k in range(self._rollout_length):
        yield self._decode_sample(unroll[j + k])

  def _decode_sample(self, sample):
    return TensorZipper.decompress(sample)


class RMFromQueue(object):
  """Replay Memory. data_queue is a list of unrolls. Each unroll has unroll_length samples,
     Each batch samples several rollouts, where each rollout has rollout_length consecutive samples."""

  def __init__(self, input_ep, output_ep, maxlen, minlen, rollout_per_unroll,
               rollout_length, batch_size, decode):
    self._maxlen = maxlen
    self._minlen = minlen
    self._rollout_length = rollout_length
    self._rollout_per_unroll = rollout_per_unroll
    self._rollout_per_batch = int(np.ceil(batch_size/float(rollout_length)))
    self._decode = decode
    self._data_queue = []
    self._next_idx = 0
    zmq_context = zmq.Context()
    pull_socket = zmq_context.socket(zmq.PULL)
    pull_socket.setsockopt(zmq.RCVHWM, 1)
    pull_socket.connect(str(input_ep, encoding = "utf-8"))
    push_socket = zmq_context.socket(zmq.PUSH)
    push_socket.setsockopt(zmq.SNDHWM, 1)
    push_socket.connect(str(output_ep, encoding = "utf-8"))
    self.pull_data_thread = Thread(target=self._pull_data,
                                   args=(pull_socket,), daemon=True)
    self.pull_data_thread.start()
    self.sample_rollout_thread = Thread(target=self._sample_rollout,
                                        args=(push_socket,), daemon=True)
    self.sample_rollout_thread.start()

  def run(self):
    self.pull_data_thread.join()
    self.sample_rollout_thread.join()

  def _pull_data(self, pull_socket):
    while True:
      idx = self._next_idx
      data = pull_socket.recv_pyobj()
      self._next_idx = (self._next_idx + 1) % self._maxlen
      if self._decode:
        data = [self._decode_sample(d) for d in data]
      if self._maxlen > len(self._data_queue):
        self._data_queue.append(data)
      else:
        self._data_queue[idx] = data

  def _sample_rollout(self, push_socket):
    while len(self._data_queue) < self._minlen:
      time.sleep(1)
    while True:
      batch_data = []
      for _ in range(self._rollout_per_batch):
        i = random.randint(0, len(self._data_queue) - 1)
        unroll = self._data_queue[i]  # backup the unroll in case of overwritten by new unroll
        j = random.randint(0, self._rollout_per_unroll - 1) * self._rollout_length
        if self._decode:
          batch_data.extend(unroll[j:j + self._rollout_length])
        else:
          batch_data.extend([self._decode_sample(d)
                             for d in unroll[j:j + self._rollout_length]])
      push_socket.send_pyobj(tuple(np.array(d) for d in zip(*batch_data)))

  def _decode_sample(self, sample):
    return TensorZipper.decompress(sample)

class ReplayMemMP(ReplayMem):
  """Replay Memory. data_queue is a list of unrolls. Each unroll has unroll_length samples,
     Each batch samples several rollouts, where each rollout has rollout_length consecutive samples."""

  def __init__(self, rm_size, unroll_length, batch_size, rollout_length,
               decode=False, n_process=1, minimal_sample=None):
    super(ReplayMemMP, self).__init__(rm_size, unroll_length, batch_size,
                                      rollout_length, decode, minimal_sample)
    self._n_process = n_process
    self._zmq_context = zmq.Context()
    self._push_socket = self._zmq_context.socket(zmq.PUSH)
    self._push_socket.setsockopt(zmq.SNDHWM, n_process)
    self._push_socket.bind("ipc:///tmp/"+str(uuid.uuid1())[:8])
    self._push_lock = Lock()
    self._input_ep = self._push_socket.getsockopt(zmq.LAST_ENDPOINT)
    self._pull_sockets = [self._zmq_context.socket(zmq.PULL) for _ in range(n_process)]
    self._output_eps = []
    for s in self._pull_sockets:
      s.setsockopt(zmq.RCVHWM, 1)
      s.bind("ipc:///tmp/"+str(uuid.uuid1())[:8])
      self._output_eps.append(s.getsockopt(zmq.LAST_ENDPOINT))
    print(self._input_ep, self._output_eps)
    self._len = 0
    self._ready = False
    self._subps = []
    # self._start_subprocess()

  def __len__(self):
    return self._len

  @staticmethod
  def _run_sub_rm(input, output, maxlen, minlen, rollout_per_unroll,
                  rollout_length, batch_size, decode):
    p = RMFromQueue(input, output, maxlen, minlen, rollout_per_unroll,
                    rollout_length, batch_size, decode)
    p.run()

  def _start_subprocess(self):
    maxl = int(np.ceil(self._maxlen / float(self._n_process)))
    minl = int(np.ceil(self._minimal_unroll / float(self._n_process)))
    self._subps = [Process(target=self._run_sub_rm,
                           args=(self._input_ep, output_ep,
                                 maxl, minl, self._rollout_per_unroll,
                                 self._rollout_length, self._batch_size,
                                 self._decode)
                           ) for output_ep in self._output_eps]
    for p in self._subps:
      p.start()

  def _stop_subprocess(self):
    for p in self._subps:
      p.terminate()

  def reset(self):
    self._stop_subprocess()
    self._len = 0
    self._ready = False
    self._start_subprocess()

  def ready_for_sample(self):
    return self._ready

  def append(self, data):
    if self._len < self._maxlen:
      self._len += 1
      if not self._ready:
        self._ready = self._len >= self._minimal_unroll
    msg = pickle.dumps(data)
    self._push_lock.acquire()
    self._push_socket.send(msg, copy=False)
    self._push_lock.release()

  def rollout_samplers(self):
    return [s.recv_pyobj for s in self._pull_sockets]

  def sample_rollout(self):
    return self._pull_sockets[0].recv_pyobj()

class ImpSampReplayMem(ReplayMem):
  """Replay Memory with important sampling. """
  def __init__(self, rm_size, unroll_length, batch_size, rollout_length, minimal_sample=None):
    super(ImpSampReplayMem, self).__init__(rm_size, unroll_length, batch_size,
                                           rollout_length, minimal_sample=minimal_sample)
    self._weights_queue = []
    self._unroll_weights = []

  def reset(self):
    super(ImpSampReplayMem, self).reset()
    self._weights_queue = []
    self._unroll_weights = []

  def append(self, data, weights):
    assert len(data) == len(weights)
    indx = self._next_idx
    self._next_idx = (self._next_idx + 1) % self._maxlen
    if self._maxlen > len(self._data_queue):
      self._unroll_weights.append(sum(weights))
      self._weights_queue.append(weights)
      self._data_queue.append(data)
      if not self._ready:
        self._ready = len(self._data_queue) >= self._minimal_unroll
    else:
      self._unroll_weights[indx] = sum(weights)
      self._weights_queue[indx] = weights
      self._data_queue[indx] = data

  def sample_rollout(self):
    curr_data_size = len(self._data_queue)
    w = self._unroll_weights[:curr_data_size]
    p = np.array(w)
    i = np.random.choice(curr_data_size, p=p/np.sum(p))
    unroll = self._data_queue[i]  # backup the unroll in case of overwritten by new unroll
    w = np.array(self._weights_queue[i])
    w = np.reshape(w, (-1, self._rollout_length))
    p = np.sum(w, axis=1)
    j = np.random.choice(self._unroll_length//self._rollout_length, p=p/np.sum(p))
    l = j * self._rollout_length
    avg_w = p[j]/self._rollout_length
    for k in range(self._rollout_length):
      yield unroll[l + k], w[j, k]/avg_w
