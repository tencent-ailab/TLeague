# TODO(pengsun): use strict message definition and message response functions
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import time
from threading import Lock

import zmq

from tleague.utils import logger
from tleague.utils.tl_types import ModelPoolErroMsg
from tleague.model_pools.model import Model


class ModelPoolAPIs(object):

  def __init__(self, model_pool_addrs):
    self._zmq_context = zmq.Context()
    self._zmq_context.setsockopt(zmq.TCP_KEEPALIVE, 1)
    self._zmq_context.setsockopt(zmq.TCP_KEEPALIVE_CNT, 60)
    self._zmq_context.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)
    self._zmq_context.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 60)
    self._req_socket = self._zmq_context.socket(zmq.REQ)
    self._pub_socket = self._zmq_context.socket(zmq.PUB)
    self.model_pool_addrs = list(model_pool_addrs)
    random.shuffle(self.model_pool_addrs)
    for addr in self.model_pool_addrs:
      ip, port1, port2 = addr.split(':')
      self._req_socket.connect("tcp://%s:%s" % (ip, port1))
      self._pub_socket.connect("tcp://%s:%s" % (ip, port2))
    self._req_lock = Lock()
    self._pub_lock = Lock()
    time.sleep(0.5)

  def check_server_set_up(self):
    for _ in self.model_pool_addrs:
      self.pull_keys()  # zmq request server in fair fashion

  def request(self, req):
    self._req_lock.acquire()
    while True:
      try:
        for msg in req[0:-1]:
          self._req_socket.send_string(msg, zmq.SNDMORE)
        self._req_socket.send_string(req[-1])
        ret = self._req_socket.recv_pyobj()
        if not isinstance(ret, ModelPoolErroMsg):
          break
        else:
          logger.log(ret.msg)  # ret isinstance ModelPoolErroMsg
        time.sleep(2)
      except BaseException as e:
        logger.error("ModelPoolAPIs may crushed on request {},"
                     " the exception:\n{}".format(req, e))
        raise e
    self._req_lock.release()
    return ret

  def pull_model(self, key):
    return self.request(["model", key])

  def pull_keys(self):
    return self.request(["keys"])

  def pull_attr(self, attr, key):
    return self.request([attr, key])

  def pull_all_attr(self, attr):
    return self.request(["all_attr", attr])

  def pull_learner_meta(self, key):
    return self.request(["learner_meta", key])

  def freeze_model(self, key):
    self._pub_lock.acquire()
    self._pub_socket.send_string('freeze', zmq.SNDMORE)
    self._pub_socket.send_string(key)
    self._pub_lock.release()

  def push_model(self, model, hyperparam, key, createtime=None,
                 freezetime=None, updatetime=None, learner_meta=None):
    self._pub_lock.acquire()
    self._pub_socket.send_string('model', zmq.SNDMORE)
    self._pub_socket.send_pyobj(
      Model(model, hyperparam, key, createtime,
            freezetime, updatetime))
    if learner_meta is not None:
      self._pub_socket.send_string('learner_meta', zmq.SNDMORE)
      self._pub_socket.send_string(key, zmq.SNDMORE)
      self._pub_socket.send_pyobj(learner_meta)
    self._pub_lock.release()
