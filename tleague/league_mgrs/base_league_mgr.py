from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from abc import ABCMeta, abstractmethod

import zmq

from tleague.model_pools.model_pool_apis import ModelPoolAPIs
from tleague.utils import logger
from tleague.utils.chkpts import ChkptsFromModelPool


class BaseLeagueMgr(metaclass=ABCMeta):

  def __init__(self, port, model_pool_addrs, restore_checkpoint_dir=None,
               save_checkpoint_root=None, save_interval_secs=3600,
               mute_actor_msg=False, save_learner_meta=False, verbose=0):
    logger.set_level(verbose)

    self._zmq_context = zmq.Context()
    self._socket = self._zmq_context.socket(zmq.REP)
    self._socket.bind("tcp://*:%s" % port)

    self._verbose = verbose
    self.mute_actor_msg = mute_actor_msg

    self._model_pool_apis = ModelPoolAPIs(model_pool_addrs)
    self._model_pool_apis.check_server_set_up()

    self._saver = ChkptsFromModelPool(self._model_pool_apis, save_learner_meta)
    self._save_checkpoint_root = save_checkpoint_root
    self._save_interval_secs = save_interval_secs

    self._learner_task_table = {}  # {learner_id: Model_instance}
    self._restore_checkpoint_dir = restore_checkpoint_dir
    self._curr_model_idx = 1

  def _gen_new_model_key(self, old_model_key):
    """ generate new model_key with "old_id_str:new_id_str" format"""
    new_model_suffix = '%04d' % self._curr_model_idx
    if old_model_key is None:
      new_model_key = 'rand_model:' + new_model_suffix
    else:
      new_model_key = old_model_key.split(':')[1] + ':' + new_model_suffix
    self._curr_model_idx += 1
    return new_model_key

  def run(self):
    if self._restore_checkpoint_dir is not None:
      self._restore_checkpoint(self._restore_checkpoint_dir)

    t_prev = time.time()
    while True:
      # should save checkpoint (block the message processing below)
      t_now = time.time()
      if (t_now - t_prev) >= self._save_interval_secs:
        checkpoint_name = "checkpoint_%s" % time.strftime('%Y%m%d%H%M%S')
        self._save_checkpoint(self._save_checkpoint_root, checkpoint_name)
        t_prev = t_now

      # message loop
      msg = self._socket.recv_string()
      logger.log("Received msg: '%s'" % msg, level=logger.DEBUG)
      if msg == 'request_actor_task':
        actor_id, learner_id = self._socket.recv_pyobj()
        actor_task = self._on_request_actor_task(actor_id, learner_id)
        self._socket.send_pyobj(actor_task)
      elif msg == 'request_learner_task':
        learner_id = self._socket.recv_pyobj()
        learner_task = self._on_request_learner_task(learner_id)
        self._socket.send_pyobj(learner_task)
      elif msg == 'query_learner_task':
        learner_id = self._socket.recv_pyobj()
        learner_task = self._on_query_learner_task(learner_id)
        self._socket.send_pyobj(learner_task)
      elif msg == 'notify_actor_task_begin':
        actor_id = self._socket.recv_pyobj()
        self._socket.send_string("ok")
        self._on_notify_actor_task_begin(actor_id)
      elif msg == 'notify_actor_task_end':
        actor_id, match_result = self._socket.recv_pyobj()
        self._socket.send_string("ok")
        self._on_notify_actor_task_end(actor_id, match_result)
      elif msg == 'notify_learner_task_begin':
        learner_id, learner_task = self._socket.recv_pyobj()
        self._socket.send_string("ok")
        self._on_notify_learner_task_begin(learner_id, learner_task)
      elif msg == 'notify_learner_task_end':
        learner_id = self._socket.recv_pyobj()
        self._socket.send_string("ok")
        self._on_notify_learner_task_end(learner_id)
      elif msg == 'request_add_model':
        model = self._socket.recv_pyobj()
        self._socket.send_string("ok")
        self._on_request_add_model(model)
      else:
        raise RuntimeError("message {} not recognized".format(msg))

  def _save_checkpoint(self, checkpoint_root, checkpoint_name):
    self._saver._save_model_checkpoint(checkpoint_root, checkpoint_name)

  def _restore_checkpoint(self, checkpoint_dir):
    max_key_idx = self._saver._restore_model_checkpoint(checkpoint_dir)
    self._curr_model_idx = max_key_idx + 1

  @abstractmethod
  def _on_request_actor_task(self, actor_id, learner_id):
    if learner_id is None:
      actor_task = self._on_request_eval_actor_task(actor_id)
    else:
      actor_task = self._on_request_train_actor_task(actor_id, learner_id)
    return actor_task

  @abstractmethod
  def _on_request_eval_actor_task(self, actor_id):
    pass

  @abstractmethod
  def _on_request_train_actor_task(self, actor_id, learner_id):
    pass

  @abstractmethod
  def _on_request_learner_task(self, learner_id):
    pass

  @abstractmethod
  def _on_query_learner_task(self, learner_id):
    pass

  def _on_notify_actor_task_begin(self, actor_id):
    pass

  @abstractmethod
  def _on_notify_actor_task_end(self, actor_id, match_result):
    pass

  def _on_notify_learner_task_begin(self, learner_id, learner_task):
    assert learner_id in self._learner_task_table
    self._learner_task_table[learner_id] = learner_task

  def _on_notify_learner_task_end(self, learner_id):
    pass

  def _on_request_add_model(self, model):
    pass
