from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from threading import Lock

import zmq

from tleague.utils.tl_types import LeagueMgrErroMsg


class LeagueMgrAPIs(object):

  def __init__(self, league_mgr_addr):
    self._zmq_context = zmq.Context()
    self._zmq_context.setsockopt(zmq.TCP_KEEPALIVE, 1)
    self._zmq_context.setsockopt(zmq.TCP_KEEPALIVE_CNT, 60)
    self._zmq_context.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)
    self._zmq_context.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 60)
    self._socket = self._zmq_context.socket(zmq.REQ)
    ip, port = league_mgr_addr.split(':')
    self._socket.connect("tcp://%s:%s" % (ip, port))
    self._req_lock = Lock()

  def request_actor_task(self, actor_id, learner_id):
    self._req_lock.acquire()
    while True:
      self._socket.send_string("request_actor_task", zmq.SNDMORE)
      self._socket.send_pyobj((actor_id, learner_id))
      task = self._socket.recv_pyobj()
      if not isinstance(task, LeagueMgrErroMsg):
        break
      time.sleep(10)
    self._req_lock.release()
    return task

  def request_learner_task(self, learner_id):
    self._req_lock.acquire()
    while True:
      self._socket.send_string("request_learner_task", zmq.SNDMORE)
      self._socket.send_pyobj(learner_id)
      task = self._socket.recv_pyobj()
      if not isinstance(task, LeagueMgrErroMsg):
        break
      time.sleep(1)
    self._req_lock.release()
    return task

  def query_learner_task(self, learner_id):
    self._req_lock.acquire()
    self._socket.send_string("query_learner_task", zmq.SNDMORE)
    self._socket.send_pyobj(learner_id)
    task = self._socket.recv_pyobj()
    self._req_lock.release()
    if isinstance(task, LeagueMgrErroMsg):
      return None
    else:
      return task

  def notify_actor_task_begin(self, actor_id):
    self._req_lock.acquire()
    self._socket.send_string("notify_actor_task_begin", zmq.SNDMORE)
    self._socket.send_pyobj(actor_id)
    assert self._socket.recv_string() == "ok"
    self._req_lock.release()

  def notify_actor_task_end(self, actor_id, match_result):
    self._req_lock.acquire()
    self._socket.send_string("notify_actor_task_end", zmq.SNDMORE)
    self._socket.send_pyobj((actor_id, match_result))
    assert self._socket.recv_string() == "ok"
    self._req_lock.release()

  def notify_learner_task_begin(self, learner_id, learner_task):
    self._req_lock.acquire()
    self._socket.send_string("notify_learner_task_begin", zmq.SNDMORE)
    self._socket.send_pyobj((learner_id, learner_task))
    assert self._socket.recv_string() == "ok"
    self._req_lock.release()

  def notify_learner_task_end(self, learner_id):
    self._req_lock.acquire()
    self._socket.send_string("notify_learner_task_end", zmq.SNDMORE)
    self._socket.send_pyobj(learner_id)
    assert self._socket.recv_string() == "ok"
    self._req_lock.release()

  def request_add_model(self, model):
    self._req_lock.acquire()
    self._socket.send_string("request_add_model", zmq.SNDMORE)
    self._socket.send_pyobj(model)
    assert self._socket.recv_string() == "ok"
    self._req_lock.release()