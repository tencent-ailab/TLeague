from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid
from abc import abstractmethod
from threading import Lock
from threading import Thread

import zmq
import pickle
from tleague.league_mgrs.league_mgr_apis import LeagueMgrAPIs
from tleague.model_pools.model_pool_apis import ModelPoolAPIs


class BaseLearner(object):
  """Base learner class.

  Define the basic workflow for a learner."""
  def __init__(self, league_mgr_addr, model_pool_addrs, learner_ports,
               learner_id=''):
    if learner_id: self._learner_id = learner_id
    else: self._learner_id = str(uuid.uuid1())

    self._zmq_context = zmq.Context()
    self._rep_socket = self._zmq_context.socket(zmq.REP)
    self._rep_socket.bind("tcp://*:%s" % learner_ports[0])
    self._pull_socket = self._zmq_context.socket(zmq.PULL)
    self._pull_socket.setsockopt(zmq.RCVHWM, 1)
    self._pull_socket.bind("tcp://*:%s" % learner_ports[1])
    self._message_thread = Thread(target=self._message_worker)
    self._message_thread.daemon = True
    self._message_thread.start()
    self._league_mgr_apis = LeagueMgrAPIs(league_mgr_addr)
    self._model_pool_apis = ModelPoolAPIs(model_pool_addrs)

    self.task = None
    self.model_key = None
    self.last_model_key = None
    self._lrn_period_count = 0  # learning period count
    self._pull_lock = Lock()

  def run(self):
    while True:
      self.task = self._request_task()
      self._init_task()
      self._train()
      self._finish_task()
      self._lrn_period_count += 1

  @abstractmethod
  def _train(self, **kwargs):
    pass

  @abstractmethod
  def _init_task(self):
    pass

  def _request_task(self):
    task = self._league_mgr_apis.request_learner_task(self._learner_id)
    self.last_model_key = self.model_key
    self.model_key = task.model_key
    # lazy freeze the model of last lp, then actors will stop the last lp.
    if self.last_model_key and self.model_key != self.last_model_key:
      self._model_pool_apis.freeze_model(self.last_model_key)
    return task

  def _query_task(self):
    task = self._league_mgr_apis.query_learner_task(self._learner_id)
    if task is not None:
      self.last_model_key = self.model_key
      self.model_key = task.model_key
    return task

  def _finish_task(self):
    self._notify_task_end()

  def _pull_data(self):
    self._pull_lock.acquire()
    data = self._pull_socket.recv(copy=False)
    self._pull_lock.release()
    return pickle.loads(data)

  def _message_worker(self):
    while True:
      msg = self._rep_socket.recv_string()
      if msg == 'learner_id':
        self._rep_socket.send_pyobj(self._learner_id)
      else:
        raise RuntimeError("message not recognized")

  def _notify_task_begin(self, task):
    self._league_mgr_apis.notify_learner_task_begin(self._learner_id, task)

  def _notify_task_end(self):
    self._league_mgr_apis.notify_learner_task_end(self._learner_id)
