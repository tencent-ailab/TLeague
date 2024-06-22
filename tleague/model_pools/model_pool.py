# TODO(pengsun): use strict message definition and message response functions
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import zmq
import pickle

from tleague.utils import logger
from tleague.model_pools.model_pool_msg import ModelPoolWriterMsg, ModelPoolReaderMsg, ModelPoolErroMsg
from tleague.utils import now


class ModelPool(object):

  def __init__(self, ports, verbose=50):
    self._model_pool = {}
    self._learner_meta = {}
    self._zmq_context = zmq.Context()
    self._rep_socket = self._zmq_context.socket(zmq.REP)
    port1, port2 = ports.split(':')
    self._rep_socket.bind("tcp://*:%s" % port1)
    self._sub_socket = self._zmq_context.socket(zmq.SUB)
    self._sub_socket.bind("tcp://*:%s" % port2)
    self._sub_socket.setsockopt_string(zmq.SUBSCRIBE, '')
    self.poll = zmq.Poller()
    self.poll.register(self._rep_socket, zmq.POLLIN)
    self.poll.register(self._sub_socket, zmq.POLLIN)

    logger.set_level(verbose)

  def run(self):
    while True:
      socks = dict(self.poll.poll())
      if socks.get(self._sub_socket) == zmq.POLLIN:
        self._write_worker()
      if socks.get(self._rep_socket) == zmq.POLLIN:
        self._read_worker()

  def _robust_pyobj_recv(self, socket):
    msg = socket.recv()
    try:
        # equivalent to recv_pyobj()
      msg = socket._deserialize(msg, pickle.loads)
    except:
      msg = None
    return msg

  def _write_worker(self):
    msg = self._robust_pyobj_recv(self._sub_socket)
    if msg == None or not isinstance(msg, ModelPoolWriterMsg):
      logger.log('Model pool msg corrupted.(Write worker)')
      return 
    if msg.model is not None:
      model = msg.model
      if model.key in self._model_pool:
        model.createtime = self._model_pool[model.key].createtime
      if model.createtime is None:
        model.createtime = model.updatetime
      self._model_pool[model.key] = model
      logger.log(now() + 'on msg write model.',
                 'key: {},'.format(model.key),
                 'create time: {}'.format(model.createtime))
    if msg.freeze is not None:  # freeze one model
      key = msg.freeze
      if key in self._model_pool:
        self._model_pool[key].freeze()
      logger.log(now() + 'on msg write freeze',
                 'key: {}'.format(key))
    if msg.learner_meta is not None:  # store learner meta data
      self._learner_meta.update(msg.learner_meta)
      logger.log(now() + 'on msg write learner_meta',
                 'key: {}'.format(list(msg.learner_meta.keys())[0]))

  def _read_worker(self):
    msg = self._robust_pyobj_recv(self._rep_socket)
    if msg == None or not isinstance(msg, ModelPoolReaderMsg):
      logger.log('Model pool msg corrupted.(Read worker)')
      try:
        self._rep_socket.send_pyobj(ModelPoolErroMsg('Msg corrupted'))
      except:
        logger.log('Read worker send_pyobj failed.')
      return 
    attr = msg.attr
    key = msg.key
    if attr == 'model':  # get one Model
      self._rep_socket.send_pyobj(
        ModelPoolErroMsg('Key {} not exists'.format(key))
        if key not in self._model_pool else self._model_pool[key]
      )
      logger.log(now() + 'on msg read model,',
                 'key: {}'.format(key))
    elif attr == 'keys':  # get all the model keys
      self._rep_socket.send_pyobj(list(self._model_pool.keys()))
      logger.log(now() + 'on msg read keys,')
    elif attr == 'all_attr':  # get the attr for all models. return {key: attr}
      self._rep_socket.send_pyobj({k: (ModelPoolErroMsg('Attribute not exists')
                                       if not hasattr(v, key) else getattr(v,
                                                                            key))
                                   for k, v in self._model_pool.items()})
      logger.log(now() + 'on msg read all_attr,')
    elif attr == 'learner_meta':  # get learner meta data
      self._rep_socket.send_pyobj(ModelPoolErroMsg('Key not exists')
                                  if key not in self._model_pool
                                  else None if key not in self._learner_meta
      else self._learner_meta[key])
      logger.log(now() + 'on msg read learner_meta')
    else:  # on pull_attr(), get the attr for one model
      # TODO(pengsun): too tricky, should use strict msg definition
      self._rep_socket.send_pyobj(
        ModelPoolErroMsg('Key not exists') if key not in self._model_pool
        else (ModelPoolErroMsg('Attribute not exists')
              if not hasattr(self._model_pool[key], attr)
              else getattr(self._model_pool[key], attr))
      )
      logger.log(now() + 'on msg read {}'.format(attr))

  def save_to_disk(self, filepath):
    raise NotImplementedError

  def load_from_disk(self, filepath):
    raise NotImplementedError
