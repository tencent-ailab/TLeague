from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from threading import Lock

import zmq

from tleague.utils.robust_socket_recv import robust_pyobj_recv
from tleague.league_mgrs.league_mgr_msg import LeagueMgrMsg, LeagueMgrErroMsg


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
      self._socket.send_pyobj(LeagueMgrMsg(attr="request_actor_task", key1=actor_id, key2=learner_id))
      task = robust_pyobj_recv(self._socket)
      if task != None and not isinstance(task, LeagueMgrErroMsg):
        break
      time.sleep(10)
    self._req_lock.release()
    return task

  def request_learner_task(self, learner_id):
    self._req_lock.acquire()
    while True:
      self._socket.send_pyobj(LeagueMgrMsg(attr="request_learner_task", key1=learner_id))
      task = robust_pyobj_recv(self._socket)
      if task != None and not isinstance(task, LeagueMgrErroMsg):
        break
      time.sleep(1)
    self._req_lock.release()
    return task

  def query_learner_task(self, learner_id):
    self._req_lock.acquire()
    self._socket.send_pyobj(LeagueMgrMsg(attr="query_learner_task", key1=learner_id))
    task = robust_pyobj_recv(self._socket)
    self._req_lock.release()
    if isinstance(task, LeagueMgrErroMsg):
      return None
    else:
      return task

  def notify_actor_task_begin(self, actor_id):
    self._req_lock.acquire()
    self._socket.send_pyobj(LeagueMgrMsg(attr="notify_actor_task_begin", key1=actor_id))
    robust_pyobj_recv(self._socket)
    self._req_lock.release()

  def notify_actor_task_end(self, actor_id, match_result):
    self._req_lock.acquire()
    self._socket.send_pyobj(LeagueMgrMsg(attr="notify_actor_task_end", key1=actor_id, key2=match_result))
    robust_pyobj_recv(self._socket)
    self._req_lock.release()

  def notify_learner_task_begin(self, learner_id, learner_task):
    self._req_lock.acquire()
    self._socket.send_pyobj(LeagueMgrMsg(attr="notify_learner_task_begin", key1=learner_id, key2=learner_task))
    robust_pyobj_recv(self._socket)
    self._req_lock.release()

  def notify_learner_task_end(self, learner_id):
    self._req_lock.acquire()
    self._socket.send_pyobj(LeagueMgrMsg(attr="notify_learner_task_end", key1=learner_id))
    robust_pyobj_recv(self._socket)
    self._req_lock.release()

  def request_add_model(self, model):
    self._req_lock.acquire()
    self._socket.send_pyobj(LeagueMgrMsg(attr="request_add_model", key1=model))
    robust_pyobj_recv(self._socket)
    self._req_lock.release()
