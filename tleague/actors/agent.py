from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import nest
from gym import spaces
import tpolicies.tp_utils as tp_utils

from tleague.utils import logger
from tleague.inference_server.api import InfServerAPIs


class PGAgent(object):
  """An agent that carries Policy Gradient compatible policy Neural Network.

  E.g., policy NNs for PPO, VTrace, etc. It only supports
  tpolicies.net_zoo.mnet_v5 or newer Neural Net.
  """

  def __init__(self, policy, ob_space, ac_space, n_v=1, scope_name="model",
               policy_config=None, use_gpu_id=-1):
    # check
    assert hasattr(policy, 'net_config_cls')
    assert hasattr(policy, 'net_build_fun')
    assert hasattr(policy, 'net_inputs_placeholders_fun')
    # bookkeeping
    self.ob_space = ob_space
    self.ob_space = ac_space
    self._ac_structure = tp_utils.template_structure_from_gym_space(ac_space)

    # build the net
    if use_gpu_id < 0:  # not using GPU
      self.sess = tf.Session()
    else:
      tf_config = tf.ConfigProto(allow_soft_placement=True,
                                 log_device_placement=True)
      tf_config.gpu_options.allow_growth = True
      self.sess = tf.Session(config=tf_config)

    policy_config = {} if policy_config is None else policy_config
    policy_config['batch_size'] = 1
    policy_config['test'] = True
    self.nc = policy.net_config_cls(ob_space, ac_space, **policy_config)
    if use_gpu_id < 0:  # not using GPU
      self.inputs_ph = policy.net_inputs_placeholders_fun(self.nc)
      self.net_out = policy.net_build_fun(self.inputs_ph, self.nc,
                                          scope=scope_name)
    else:
      with tf.device('/gpu:{}'.format(use_gpu_id)):
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

    # numpy rnn state stuff (if any)
    if self.net_out.S is None:
      self._hs_len = None
      self._state = None
    else:
      self._hs_len = self.net_out.S.shape[1].value
      self._state = np.zeros(shape=(self._hs_len,), dtype=np.float32)
    pass

  @property
  def state(self):
    # rnn state
    return self._state

  def load_model(self, np_params):
    self.sess.run(self.params_assign_ops[:len(np_params)], feed_dict={
      ph: np_p for ph, np_p in zip(self.params_ph, np_params)})

  def reset(self, obs=None):
    if self._hs_len is None:
      return
    self._state = np.zeros(shape=(self._hs_len,), dtype=np.float32)

  def _feed_obs(self, obs):
    """prepare feed_dict from input obs"""
    # NOTE1: safe to nest.flatten even an OrderedDict, in which case .X and obs
    #   are guaranteed to have exactly the same keys leading to the same flattening
    #   order!
    # NOTE2: the weired-looking [ob_np] is a quick hacking for native python
    #   types (int32, float, etc.), which in effect inserts a leading
    #   batch_size = 1 dim.
    feed_dict = {ob_ph: [ob_np] for ob_ph, ob_np
                 in zip(nest.flatten(self.inputs_ph.X), nest.flatten(obs))}
    if self._state is not None:
      feed_dict[self.inputs_ph.S] = [self._state]
      # always one-step, non-terminal
      feed_dict[self.inputs_ph.M] = [np.zeros(shape=())]
    return feed_dict

  def _forward(self, obs, fetches, action=None):
    feed_dict = self._feed_obs(obs)
    if self._state is not None:
      fetches['state'] = self.net_out.S
    if action is not None:
      for ac_ph, ac_np in zip(nest.flatten(self.inputs_ph.A), nest.flatten(action)):
        feed_dict[ac_ph] = [ac_np]

    ret = self.sess.run(fetches, feed_dict=feed_dict)
    # NOTE: do not squeeze the batch_size dim here!

    self._last_state = self._state
    self._state = (None if self._state is None else
                   _squeeze_batch_size_singleton_dim(ret['state']))
    return ret

  def forward_squeezed(self, obs):
    # prepare fetches dict
    fetches = {
      'a': nest.map_structure_up_to(self._ac_structure, lambda head: head.sam,
                                    self.net_out.self_fed_heads),
      'neglogp': nest.map_structure_up_to(self._ac_structure,
                                          lambda head: head.neglogp,
                                          self.net_out.self_fed_heads),
      # 'logits': nest.map_structure_up_to(self._ac_structure,
      #                                    lambda head: head.logits,
      #                                    self.net_out.self_fed_heads),
      'v': self.net_out.value_head if self.net_out.value_head is not None else []
    }
    ret = self._forward(obs, fetches)
    a = _squeeze_batch_size_singleton_dim(ret['a'])
    v = _squeeze_batch_size_singleton_dim(ret['v'])
    neglogp = _squeeze_batch_size_singleton_dim(ret['neglogp'])
    # logits = _squeeze_batch_size_singleton_dim(ret['logits'])
    return a, v, self._last_state, neglogp

  def step(self, obs):
    # prepare fetches dict
    fetches = {
      'a': nest.map_structure_up_to(self._ac_structure, lambda head: head.sam,
                                    self.net_out.self_fed_heads),
    }
    ret = self._forward(obs, fetches)
    return _squeeze_batch_size_singleton_dim(ret['a'])

  def step_logits(self, obs):
    fetches = {
      'a': nest.map_structure_up_to(self._ac_structure, lambda head: head.sam,
                                    self.net_out.self_fed_heads),
      'logits': nest.map_structure_up_to(self._ac_structure,
                                         lambda head: head.logits,
                                         self.net_out.self_fed_heads)
    }
    ret = self._forward(obs, fetches)
    return _squeeze_batch_size_singleton_dim(ret['a']), ret['logits']

  def update_state(self, obs):
    fetches = {}
    self._forward(obs, fetches)
    return self._state

  def logits(self, obs, action=None):
    if action is None:
      assert self.net_out.self_fed_heads is not None
      heads = self.net_out.self_fed_heads
    else:
      assert self.net_out.outer_fed_heads is not None
      heads = self.net_out.outer_fed_heads
    fetches = {'logits': nest.map_structure_up_to(self._ac_structure,
                                                  lambda head: head.logits,
                                                  heads)}
    ret = self._forward(obs, fetches, action)
    return _squeeze_batch_size_singleton_dim(ret['logits'])


class PGAgentGPU(object):
  """An agent that uses (remote) GPU Inference. The same usage of PGAgent
  """

  def __init__(self, infserver_addr, ds, hs_len=None, compress=True):
    self.apis = InfServerAPIs(infserver_addr, ds, compress)
    self.ds = ds
    self._hs_len = hs_len
    # numpy rnn state stuff (if any)
    if self._hs_len is None:
      self._state = None
    else:
      self._state = np.zeros(shape=(self._hs_len,), dtype=np.float32)

  @property
  def state(self):
    # rnn state
    return self._state

  def load_model(self, np_params):
    pass

  def reset(self, obs=None):
    if self._hs_len is None:
      return
    self._state = np.zeros(shape=(self._hs_len,), dtype=np.float32)

  def _forward(self, obs, action=None):
    data = [obs]
    if action is not None:
      data.append(action)
    if self._state is not None:
      data.extend([self.state, np.array(False)])
    ret = self.apis.request_output(self.ds.structure(data))
    if self._state is not None:
      self._last_state = self._state
      self._state = ret['state']
    return ret

  def forward_squeezed(self, obs):
    ret = self._forward(obs)
    return ret['a'], ret.get('v', []), self._last_state, ret['neglogp']

  def step(self, obs):
    # prepare fetches dict
    ret = self._forward(obs)
    return ret['a']

  def update_state(self, obs):
    self._forward(obs)
    return self._state

  def logits(self, obs, action=None):
    ret = self._forward(obs, action)
    return ret['logits']


class PPOAgent(object):
  """An agent that carries PPO compatible policy Neural Network. Obsolete, use
   PGAgent instead."""

  def __init__(self, policy, ob_space, ac_space, n_v=1,
               scope_name="model", policy_config=None):
    sess = tf.Session()
    self.ob_space = ob_space
    self.ac_space = ac_space
    policy_config = {} if policy_config is None else policy_config

    def create_policy():
      return policy(ob_space, ac_space, n_v=n_v, nbatch=1,
                    scope_name=scope_name, **policy_config)

    with sess.as_default():
      if 'use_xla' in policy_config and policy_config['use_xla']:
        try:
          with tf.xla.experimental.jit_scope(True):
            self.policy = create_policy()
        except:
          logger.warn("using tf.xla in PPOAgent requires tf version>=1.15.")
          self.policy = create_policy()
      else:
        self.policy = create_policy()
    self._state = self.policy.initial_state
    params = tf.trainable_variables(scope=scope_name)
    new_params = [tf.placeholder(p.dtype, shape=p.get_shape()) for p in params]
    param_assign_ops = [p.assign(new_p) for p, new_p in zip(params, new_params)]
    tf.global_variables_initializer().run(session=sess)

    def load_model(loaded_params):
      sess.run(param_assign_ops[:len(loaded_params)],
               feed_dict={p: v for p, v in zip(new_params, loaded_params)})

    self.load_model = load_model

  @property
  def state(self):
    return self._state

  def reset(self, obs=None):
    self._state = self.policy.initial_state  # used to store rnn hidden layer

  def _feed_obs(self, obs):
    if isinstance(self.ob_space, spaces.Tuple):
      feed_obs = [[ob] for ob in obs]
    else:
      feed_obs = [obs]
    return feed_obs

  def _forward(self, obs):
    if self._state is not None:
      a, v, state, p, logits = self.policy.step(self._feed_obs(obs),
                                                [self._state])
      self._last_state = self._state
      self._state = state
    else:
      a, v, state, p, logits = self.policy.step(self._feed_obs(obs))
      self._last_state = self._state
    return a, v, state, p, logits

  def forward_squeezed(self, obs):
    a, v, state, p, logits = self._forward(obs)
    if isinstance(a, list):  # action from spaces.Tuple
      return [b.squeeze() for b in
              a], v.squeeze(), self._last_state, p.squeeze()
    else:
      return a.squeeze(), v.squeeze(), self._last_state, p.squeeze()

  def step(self, obs):
    a, v, state, p, logits = self._forward(obs)
    if isinstance(a, list):
      return [b.squeeze() for b in a]
    else:
      return a.squeeze()

  def update_state(self, obs):
    if hasattr(self.policy, 'next_state'):
      state = self.policy.next_state(self._feed_obs(obs), [self._state])
      self._last_state = self._state
      self._state = state
    else:
      a, v, state, p, logits = self._forward(obs)
    return state

  def logits(self, obs, action=None):
    if action is None:
      a, v, state, p, logits = self._forward(obs)
      return logits.squeeze()
    else:
      if isinstance(self.ob_space, spaces.Tuple):
        feed_obs = [[ob] for ob in obs]
      else:
        feed_obs = [obs]
      if isinstance(self.ac_space, spaces.Tuple):
        feed_ac = [[ac] for ac in action]
      else:
        feed_ac = [action]

      if self._state is not None:
        logits, state = self.policy.run_logits(feed_obs, feed_ac, [self._state])
        self._last_state = self._state
        self._state = state
      else:
        logits = self.policy.run_logits(feed_obs, feed_ac)
        self._last_state = self._state
    return logits.squeeze()


PPOAgent2 = PGAgent
"""PPOAgent2 here for backwards class name compatibility."""


def _squeeze_batch_size_singleton_dim(st):
  return nest.map_structure(
    lambda x: np.squeeze(x, axis=0) if isinstance(x, np.ndarray) else x, st)


def _insert_batch_size_singleton_dim(st):
  return nest.map_structure(
    lambda x: np.expand_dims(x, axis=0) if isinstance(x, np.ndarray) else x, st)
