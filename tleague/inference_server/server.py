import pickle
import time
import os
from queue import Queue
from threading import Thread, Lock

import zmq
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import nest
import tpolicies.tp_utils as tp_utils

from tleague.model_pools.model_pool_apis import ModelPoolAPIs
from tleague.league_mgrs.league_mgr_apis import LeagueMgrAPIs
from tleague.utils.io import TensorZipper


class InferDataServer(object):

  def __init__(self, port, batch_size, ds, batch_worker_num=4,
               use_gpu=True, compress=False):
    self._zmq_context = zmq.Context()
    self._rep_socket = self._zmq_context.socket(zmq.ROUTER)
    self._rep_socket.bind("tcp://*:%s" % port)
    self._batch_size = batch_size
    self._compress = compress
    self._data_queue = Queue(batch_size) # data from actor
    self._ready = False
    self._response_queue = Queue()
    self._pull_data_thread = Thread(target=self._pull_data, daemon=True)
    self._pull_data_thread.start()
    shapes, dtypes = list(zip(*ds.flatten_spec, ([], tf.string)))
    dataset = tf.data.Dataset.range(batch_worker_num).apply(
        tf.contrib.data.parallel_interleave(
          lambda x: tf.data.Dataset.from_generator(
            self.data_generator, dtypes, shapes),
          cycle_length=batch_worker_num,
          sloppy=True,
          buffer_output_elements=int(batch_size/batch_worker_num))).apply(
                  tf.contrib.data.batch_and_drop_remainder(batch_size))
    if use_gpu:
      prefetch_op = tf.contrib.data.prefetch_to_device(
        device="/gpu:0", buffer_size=1)
      dataset = dataset.apply(prefetch_op)
    else:
      dataset = dataset.prefetch(buffer_size=1)
    batch = dataset.make_one_shot_iterator().get_next()
    self._batch_input = ds.make_structure(batch[:-1])
    self._batch_data_id = batch[-1]

  @property
  def ready(self):
    if not self._ready:
      self._ready = self._data_queue.qsize() >= self._batch_size
    return self._ready

  def data_generator(self):
    while True:
      while True:
        try:
          msg = self._data_queue.get_nowait()
          break
        except:
          time.sleep(0.01)
      if self._compress:
        data = TensorZipper.decompress(msg[-1])
      else:
        data = pickle.loads(msg[-1])
      yield data + (msg[0].bytes,)

  def _pull_data(self):
    def _squeeze_batch_size_singleton_dim(st):
      return nest.map_structure(
        lambda x: np.squeeze(x, axis=0) if isinstance(x, np.ndarray) else x, st)
    while True:
      # pull obs
      try:
        msg = self._rep_socket.recv_multipart(zmq.NOBLOCK, copy=False)
        self._data_queue.put(msg)
      except:
        pass
      # send response
      while not self._response_queue.empty():
        ids, outputs = self._response_queue.get()
        outputs = [_squeeze_batch_size_singleton_dim(o) for o in outputs]
        for data_id, output in zip(ids, outputs):
          self._rep_socket.send_multipart([data_id, b'', pickle.dumps(output)])

  def response(self, ids, outputs):
    self._response_queue.put((ids, outputs))

class InfServer(object):
  def __init__(self, league_mgr_addr, model_pool_addrs, port, ds,
               batch_size, ob_space, ac_space, policy, outputs=['a'],
               policy_config={}, gpu_id=0, compress=True,
               batch_worker_num=4, update_model_seconds=60,
               learner_id=None, log_seconds=60, model_key="", **kwargs):
    self._update_model_seconds = update_model_seconds
    self._log_seconds = log_seconds
    self._learner_id = learner_id
    if model_key:
      self._league_mgr_apis = None
      self.is_rl = False
      self.model_key = model_key
    else:
      self._league_mgr_apis = LeagueMgrAPIs(league_mgr_addr)
      self.is_rl = True
      self.model_key = None
    self.model = None
    self._model_pool_apis = ModelPoolAPIs(model_pool_addrs)
    assert hasattr(policy, 'net_config_cls')
    assert hasattr(policy, 'net_build_fun')
    # bookkeeping
    self.ob_space = ob_space
    self.ob_space = ac_space
    self.batch_size = batch_size
    self._ac_structure = tp_utils.template_structure_from_gym_space(ac_space)
    self.outputs = outputs
    # build the net
    policy_config = {} if policy_config is None else policy_config
    policy_config['batch_size'] = batch_size
    use_gpu = (gpu_id >= 0)
    self.data_server = InferDataServer(
      port=port,
      batch_size=batch_size,
      ds=ds,
      batch_worker_num=batch_worker_num,
      use_gpu=use_gpu,
      compress=compress,
    )
    config = tf.ConfigProto(allow_soft_placement=True)
    if use_gpu:
      config.gpu_options.visible_device_list = str(gpu_id)
      config.gpu_options.allow_growth = True
      if 'use_xla' in policy_config and policy_config['use_xla']:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    self._sess = tf.Session(config=config)
    self.nc = policy.net_config_cls(ob_space, ac_space, **policy_config)
    self.net_out = policy.net_build_fun(self.data_server._batch_input, self.nc,
                                        scope='Inf_server')
    # saving/loading ops
    self.params = self.net_out.vars.all_vars
    self.params_ph = [tf.placeholder(p.dtype, shape=p.get_shape())
                      for p in self.params]
    self.params_assign_ops = [
      p.assign(np_p) for p, np_p in zip(self.params, self.params_ph)
    ]
    # initialize the net params
    tf.global_variables_initializer().run(session=self._sess)
    self.setup_fetches(outputs)
    self.id_and_fetches = [self.data_server._batch_data_id, self.fetches]
    self._update_model()

  def load_model(self, loaded_params):
    self._sess.run(self.params_assign_ops[:len(loaded_params)],
                   feed_dict={p: v for p, v in zip(self.params_ph, loaded_params)})

  def setup_fetches(self, outputs):
    def split_batch(template, tf_structure):
      split_flatten = zip(*[tf.split(t, self.batch_size)
                            for t in nest.flatten_up_to(template, tf_structure)])
      return [nest.pack_sequence_as(template, flatten) for flatten in split_flatten]

    if self.nc.use_self_fed_heads:
      a = nest.map_structure_up_to(self._ac_structure, lambda head: head.sam,
                                   self.net_out.self_fed_heads)
      neglogp = nest.map_structure_up_to(self._ac_structure,
                                         lambda head: head.neglogp,
                                         self.net_out.self_fed_heads)
      flatparam = nest.map_structure_up_to(self._ac_structure,
                                           lambda head: head.flatparam,
                                           self.net_out.self_fed_heads)
      self.all_outputs = {
        'a': split_batch(self._ac_structure, a),
        'neglogp': split_batch(self._ac_structure, neglogp),
        'flatparam': split_batch(self._ac_structure, flatparam),
        'v': tf.split(self.net_out.value_head, self.batch_size)
            if self.net_out.value_head is not None else [[]]*self.batch_size,
        'state': tf.split(self.net_out.S, self.batch_size)
            if self.net_out.S is not None else [[]]*self.batch_size
      }
    else:
      flatparam = nest.map_structure_up_to(self._ac_structure,
                                           lambda head: head.flatparam,
                                           self.net_out.outer_fed_heads)
      self.all_outputs = {
        'flatparam': split_batch(self._ac_structure, flatparam),
        'state': tf.split(self.net_out.S, self.batch_size)
            if self.net_out.S is not None else [[]] * self.batch_size
      }
    if self.nc.use_lstm and 'state' not in outputs:
      outputs.append('state')
    self.fetches = [dict(zip(outputs, pred))
                    for pred in zip(*[self.all_outputs[o] for o in outputs])]

  def _update_model(self):
    if self.is_rl:
      if (self.model_key is None or
          (self.model is not None and self.model.is_freezed())):
        self._query_task()
    if self._should_update_model(self.model, self.model_key):
      self.model = self._model_pool_apis.pull_model(self.model_key)
      self.load_model(self.model.model)

  def _query_task(self):
    assert self.is_rl, '_query_task can be use in RL!'
    task = self._league_mgr_apis.query_learner_task(self._learner_id)
    while task is None:
      print('Learner has not request task! wait...')
      time.sleep(5)
      task = self._league_mgr_apis.query_learner_task(self._learner_id)
    self.last_model_key = self.model_key
    self.model_key = task.model_key
    return task

  def _should_update_model(self, model, model_key):
    if model is None or model_key != model.key:
      return True
    elif model.is_freezed():
      return False
    else:
      return self._model_pool_apis.pull_attr('updatetime', model_key) > model.updatetime

  def run(self):
    while not self.data_server.ready:
      time.sleep(10)
      print('Waiting at least {} actors to '
            'connect ...'.format(self.batch_size), flush=True)
    last_update_time = time.time()
    last_log_time = last_update_time
    batch_num = 0
    last_log_batch_num = 0
    pid = os.getpid()
    while True:
      # input is pre-fetched in self.data_server
      data_ids, outputs = self._sess.run(self.id_and_fetches, {})
      self.data_server.response(data_ids, outputs)
      batch_num += 1
      t0 = time.time()
      if t0 > last_update_time + self._update_model_seconds:
        self._update_model()
        last_update_time = t0
      t0 = time.time()
      if t0 > last_log_time + self._log_seconds:
        cost = t0 - last_log_time
        sam_num = self.batch_size * (batch_num - last_log_batch_num)
        print('Process {} predicts {} samples costs {} seconds, fps {}'.format(
          pid, sam_num, cost, sam_num / cost), flush=True)
        last_log_batch_num = batch_num
        last_log_time = t0
