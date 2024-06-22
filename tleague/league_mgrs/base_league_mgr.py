from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from abc import ABCMeta, abstractmethod
from threading import Thread
import zmq

from tleague.model_pools.model_pool_apis import ModelPoolAPIs
from tleague.utils import logger
from tleague.utils.chkpts import ChkptsFromModelPool
from tleague.utils.robust_socket_recv import robust_pyobj_recv, robust_string_recv
from tleague.league_mgrs.league_mgr_msg import LeagueMgrMsg, LeagueMgrErroMsg, LeagueMgrOKMsg


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

    self._save_thread = Thread(target=self._save_checkpoint_thread)
    self._save_thread.daemon = True

  def _save_checkpoint_thread(self):
    t_prev = time.time()
    while True:
      t_now = time.time()
      if (t_now - t_prev) >= self._save_interval_secs:
        checkpoint_name = "checkpoint_%s" % time.strftime('%Y%m%d%H%M%S')
        self._save_checkpoint(self._save_checkpoint_root, checkpoint_name)
        t_prev = t_now
      time.sleep(5)

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
    self._save_thread.start()

    while True:
      msg = robust_pyobj_recv(self._socket)
      if msg == None or not isinstance(msg, LeagueMgrMsg):
        logger.log('LeagueMgr msg corrupted.')
        self._socket.send_pyobj(LeagueMgrErroMsg(msg='Msg corrupted'))
        continue
      attr = msg.attr
      logger.log("Received msg: '%s'" % attr, level=logger.DEBUG)
      if attr == 'request_actor_task':
        actor_id, learner_id = msg.key1, msg.key2
        actor_task = self._on_request_actor_task(actor_id, learner_id)
        self._socket.send_pyobj(actor_task)
      elif attr == 'request_learner_task':
        learner_id = msg.key1
        learner_task = self._on_request_learner_task(learner_id)
        self._socket.send_pyobj(learner_task)
      elif attr == 'query_learner_task':
        learner_id = msg.key1
        learner_task = self._on_query_learner_task(learner_id)
        self._socket.send_pyobj(learner_task)
      elif attr == 'notify_actor_task_begin':
        actor_id = msg.key1
        self._socket.send_pyobj(LeagueMgrOKMsg(msg="ok"))
        self._on_notify_actor_task_begin(actor_id)
      elif attr == 'notify_actor_task_end':
        actor_id, match_result = msg.key1, msg.key2
        self._socket.send_pyobj(LeagueMgrOKMsg(msg="ok"))
        self._on_notify_actor_task_end(actor_id, match_result)
      elif attr == 'notify_learner_task_begin':
        learner_id, learner_task = msg.key1, msg.key2
        self._socket.send_pyobj(LeagueMgrOKMsg(msg="ok"))
        self._on_notify_learner_task_begin(learner_id, learner_task)
      elif attr == 'notify_learner_task_end':
        learner_id = msg.key1
        self._socket.send_pyobj(LeagueMgrOKMsg(msg="ok"))
        self._on_notify_learner_task_end(learner_id)
      elif attr == 'request_add_model':
        model = msg.key1
        self._socket.send_pyobj(LeagueMgrOKMsg(msg="ok"))
        self._on_request_add_model(model)
      else:
        self._socket.send_pyobj(LeagueMgrErroMsg(msg="Unrecognized string.(League Mgr)"))

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
