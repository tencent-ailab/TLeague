# TODO(pengsun): use strict message definition and message response functions
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import time
from threading import Lock

import zmq
import pickle

from tleague.utils import logger
from tleague.utils.robust_socket_recv import robust_pyobj_recv
from tleague.model_pools.model import Model
from tleague.model_pools.model_pool_msg import ModelPoolWriterMsg, ModelPoolReaderMsg, ModelPoolErroMsg


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
    attr = req[0]
    key = req[1] if len(req) == 2 else None
    self._req_lock.acquire()
    while True:
      try:
        self._req_socket.send_pyobj(ModelPoolReaderMsg(attr=attr, key=key))
        ret = robust_pyobj_recv(self._req_socket)
        if isinstance(ret, ModelPoolErroMsg):
          logger.log(ret.msg) # ret isinstance ModelPoolErroMsg
        else:
          break 
        time.sleep(2)
      except BaseException as e:
        logger.error("error: ModelPoolAPIs may crushed on request {},"
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
    self._pub_socket.send_pyobj(ModelPoolWriterMsg(freeze=key))
    self._pub_lock.release()

  def push_model(self, model, hyperparam, key, createtime=None,
                 freezetime=None, updatetime=None, learner_meta=None):
    self._pub_lock.acquire()
    model = Model(model, hyperparam, key, createtime,
            freezetime, updatetime)
    if learner_meta is not None:
      learner_meta = {key: learner_meta}
    self._pub_socket.send_pyobj(
      ModelPoolWriterMsg(model=model, learner_meta=learner_meta))
    self._pub_lock.release()
