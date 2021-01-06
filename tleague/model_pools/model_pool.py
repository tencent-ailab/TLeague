# TODO(pengsun): use strict message definition and message response functions
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import zmq

from tleague.utils import logger
from tleague.utils.tl_types import ModelPoolErroMsg
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

  def _write_worker(self):
    msg = self._sub_socket.recv_string()
    if msg == 'model':
      model = self._sub_socket.recv_pyobj()
      if model.key in self._model_pool:
        model.createtime = self._model_pool[model.key].createtime
      if model.createtime is None:
        model.createtime = model.updatetime
      self._model_pool[model.key] = model
      logger.log(now() + 'on msg write model.',
                 'key: {},'.format(model.key),
                 'create time: {}'.format(model.createtime))
    elif msg == 'freeze':  # freeze one model
      key = self._sub_socket.recv_string()
      if key in self._model_pool:
        self._model_pool[key].freeze()
      logger.log(now() + 'on msg write freeze',
                 'key: {}'.format(key))
    elif msg == 'learner_meta':  # store learner meta data
      key = self._sub_socket.recv_string()
      learner_meta = self._sub_socket.recv_pyobj()
      self._learner_meta[key] = learner_meta
      logger.log(now() + 'on msg write learner_meta',
                 'key: {}'.format(key))
    else:
      raise RuntimeError("message {} not recognized".format(msg))

  def _read_worker(self):
    msg = self._rep_socket.recv_string()
    if msg == 'model':  # get one Model
      key = self._rep_socket.recv_string()
      self._rep_socket.send_pyobj(
        ModelPoolErroMsg('Key {} not exits'.format(key))
        if key not in self._model_pool else self._model_pool[key]
      )
      logger.log(now() + 'on msg read model,',
                 'key: {}'.format(key))
    elif msg == 'keys':  # get all the model keys
      self._rep_socket.send_pyobj(list(self._model_pool.keys()))
      logger.log(now() + 'on msg read keys,')
    elif msg == 'all_attr':  # get the attr for all models. return {key: attr}
      attr = self._rep_socket.recv_string()
      self._rep_socket.send_pyobj({k: (ModelPoolErroMsg('Attribute not exits')
                                       if not hasattr(v, attr) else getattr(v,
                                                                            attr))
                                   for k, v in self._model_pool.items()})
      logger.log(now() + 'on msg read all_attr,')
    elif msg == 'learner_meta':  # get learner meta data
      key = self._rep_socket.recv_string()
      self._rep_socket.send_pyobj(ModelPoolErroMsg('Key not exits')
                                  if key not in self._model_pool
                                  else None if key not in self._learner_meta
      else self._learner_meta[key])
      logger.log(now() + 'on msg read learner_meta')
    else:  # on pull_attr(), get the attr for one model
      # TODO(pengsun): too tricky, should use strict msg definition
      attr = msg
      key = self._rep_socket.recv_string()
      self._rep_socket.send_pyobj(
        ModelPoolErroMsg('Key not exits') if key not in self._model_pool
        else (ModelPoolErroMsg('Attribute not exits')
              if not hasattr(self._model_pool[key], attr)
              else getattr(self._model_pool[key], attr))
      )
      logger.log(now() + 'on msg read {}'.format(msg))

  def save_to_disk(self, filepath):
    raise NotImplementedError

  def load_from_disk(self, filepath):
    raise NotImplementedError
