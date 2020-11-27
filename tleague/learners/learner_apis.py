from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from threading import Lock

import zmq


class LearnerAPIs(object):

  def __init__(self, learner_addr):
    self._zmq_context = zmq.Context()
    self._zmq_context.setsockopt(zmq.TCP_KEEPALIVE, 1)
    self._zmq_context.setsockopt(zmq.TCP_KEEPALIVE_CNT, 60)
    self._zmq_context.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)
    self._zmq_context.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 60)
    ip, port_req, port_push = learner_addr.split(':')
    self._push_socket = self._zmq_context.socket(zmq.PUSH)
    self._push_socket.setsockopt(zmq.SNDHWM, 1)
    self._push_socket.connect("tcp://%s:%s" % (ip, port_push))
    self._req_socket = self._zmq_context.socket(zmq.REQ)
    self._req_socket.connect("tcp://%s:%s" % (ip, port_req))
    self._req_lock = Lock()
    self._push_lock = Lock()
    time.sleep(0.5)

  def push_data(self, data):
    self._push_lock.acquire()
    self._push_socket.send_pyobj(data)
    self._push_lock.release()

  def request_learner_id(self):
    self._req_lock.acquire()
    self._req_socket.send_string("learner_id")
    learner_id = self._req_socket.recv_pyobj()
    self._req_lock.release()
    return learner_id


class ImLearnerAPIs(LearnerAPIs):

  def request_replay_task(self):
    self._req_lock.acquire()
    self._req_socket.send_string("replay_task")
    replay_task = self._req_socket.recv_pyobj()
    self._req_lock.release()
    return replay_task