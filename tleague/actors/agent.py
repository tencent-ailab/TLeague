from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import nest
import tpolicies.tp_utils as tp_utils

from tleague.utils.data_structure import InfData
from tleague.inference_server.api import InfServerAPIs


class Agent(object):
  def __init__(self, policy, ob_space, ac_space, n_v=1, scope_name="model",
               policy_config=None, use_gpu_id=-1, infserver_addr=None,
               compress=True):
    # check
    assert hasattr(policy, 'net_config_cls')
    assert hasattr(policy, 'net_build_fun')
    assert hasattr(policy, 'net_inputs_placeholders_fun')

    # bookkeeping
    self.ob_space = ob_space
    self.ob_space = ac_space
    self._ac_structure = tp_utils.template_structure_from_gym_space(ac_space)
    self.infserver_addr = infserver_addr
    self.compress = compress  # send compressed data to infserver
    self.n_v = n_v  # number of reward channels

    policy_config = {} if policy_config is None else policy_config
    policy_config['batch_size'] = 1
    policy_config['test'] = True
    self.nc = policy.net_config_cls(ob_space, ac_space, **policy_config)
    self.rnn = (False if 'use_lstm' not in policy_config
                else policy_config['use_lstm'])
    # numpy rnn state stuff (if any)
    if not self.rnn:
      self._hs_len = None
      self._state = None
    else:
      self._hs_len = self.nc.hs_len
      self._state = np.zeros(shape=(self._hs_len,), dtype=np.float32)

    if infserver_addr is None:
      # build the net
      if use_gpu_id < 0:  # not using GPU
        self.sess = tf.Session()
        device = '/cpu:0'
      else:
        device = '/gpu:{}'.format(use_gpu_id)
        tf_config = tf.ConfigProto(allow_soft_placement=True,
                                   log_device_placement=True)
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)
      with tf.device(device):
        self.inputs_ph = policy.net_inputs_placeholders_fun(self.nc)
        self.net_out = policy.net_build_fun(self.inputs_ph, self.nc,
                                            scope=scope_name)
      # saving/loading ops
      self.params = self.net_out.vars.all_vars
      self.params_ph = [tf.placeholder(p.dtype, shape=p.get_shape())
                        for p in self.params]
      self.params_assign_ops = [
        p.assign(np_p) for p, np_p in zip(self.params, self.params_ph)
      ]

      # initialize the net params
      tf.global_variables_initializer().run(session=self.sess)
    else:
      ds = InfData(ob_space, ac_space, policy_config['use_self_fed_heads'],
                   self.rnn, self._hs_len)
      self.apis = InfServerAPIs(infserver_addr, ds, compress)
      self.ds = ds

  @property
  def state(self):
    # rnn state
    return self._state

  def load_model(self, np_params):
    if self.infserver_addr is None:
      self.sess.run(self.params_assign_ops[:len(np_params)], feed_dict={
        ph: np_p for ph, np_p in zip(self.params_ph, np_params)})

  def reset(self, obs=None):
    if self._hs_len is None:
      return
    self._state = np.zeros(shape=(self._hs_len,), dtype=np.float32)

  def _forward(self, obs, fetches=None, action=None):
    if self.infserver_addr is None:
      def _feed_obs(obs, action):
        """prepare feed_dict from input obs"""
        feed_dict = {ob_ph: [ob_np] for ob_ph, ob_np
                     in zip(nest.flatten(self.inputs_ph.X), nest.flatten(obs))}
        if self._state is not None:
          feed_dict[self.inputs_ph.S] = [self._state]
          # always one-step, non-terminal
          feed_dict[self.inputs_ph.M] = [np.zeros(shape=())]
        if action is not None:
          assert self.inputs_ph.A is not None
          for ac_ph, ac_np in zip(nest.flatten(self.inputs_ph.A),
                                  nest.flatten(action)):
            feed_dict[ac_ph] = [ac_np]
        return feed_dict
      fetches = {} if fetches is None else fetches
      feed_dict = _feed_obs(obs, action)
      if self._state is not None:
        fetches['state'] = self.net_out.S
      ret = self.sess.run(fetches, feed_dict=feed_dict)
      ret = dict([(k, _squeeze_batch_size_singleton_dim(v))
                  for k, v in ret.items()])
    else:
      data = [obs]
      if action is not None:
        data.append(action)
      if self._state is not None:
        data.extend([self.state, np.array(False)])
      ret = self.apis.request_output(self.ds.structure(data))
    if self._state is not None:
      self._last_state = self._state
      self._state = ret.pop('state')
    return ret

  def forward_squeezed(self, obs):
    # return action, other params for rl
    raise NotImplementedError

  def step(self, obs):
    # return action
    raise NotImplementedError


class PGAgent(Agent):
  """An agent that carries Policy Gradient compatible policy Neural Network.

  E.g., policy NNs for PPO, VTrace, etc. It only supports
  tpolicies.net_zoo.mnet_v5 or newer Neural Net.
  """
  def forward_squeezed(self, obs):
    if self.infserver_addr is None:
      # prepare fetches dict
      fetches = {
        'a': nest.map_structure_up_to(self._ac_structure, lambda head: head.sam,
                                      self.net_out.self_fed_heads),
        'neglogp': nest.map_structure_up_to(self._ac_structure,
                                            lambda head: head.neglogp,
                                            self.net_out.self_fed_heads),
        'v': self.net_out.value_head if self.net_out.value_head is not None else []
      }
    else:
      fetches = None
    ret = self._forward(obs, fetches=fetches)
    return ret['a'], ret.get('v', []), self._last_state, ret['neglogp']

  def step(self, obs):
    if self.infserver_addr is None:
      # prepare fetches dict
      fetches = {
        'a': nest.map_structure_up_to(self._ac_structure, lambda head: head.sam,
                                      self.net_out.self_fed_heads),
      }
    else:
      fetches = None
    ret = self._forward(obs, fetches=fetches)
    return ret['a']

  def update_state(self, obs):
    fetches = {}
    self._forward(obs, fetches=fetches)
    return self._state

  def logits(self, obs, action=None):
    if self.infserver_addr is None:
      if action is None:
        assert self.net_out.self_fed_heads is not None
        heads = self.net_out.self_fed_heads
      else:
        assert self.net_out.outer_fed_heads is not None
        heads = self.net_out.outer_fed_heads
      fetches = {'logits': nest.map_structure_up_to(self._ac_structure,
                                                    lambda head: head.logits,
                                                    heads)}
    else:
      fetches = None
    ret = self._forward(obs, fetches, action)
    return ret['logits']


def _squeeze_batch_size_singleton_dim(st):
  return nest.map_structure(
    lambda x: np.squeeze(x, axis=0) if isinstance(x, np.ndarray) else x, st)


def _insert_batch_size_singleton_dim(st):
  return nest.map_structure(
    lambda x: np.expand_dims(x, axis=0) if isinstance(x, np.ndarray) else x, st)
