import os
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tpolicies.tp_utils as tp_utils
from tensorflow.contrib.framework import nest
from tleague.inference_server.api import InfServerAPIs
from tleague.utils.data_structure import InfData


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
    self.ac_space = ac_space
    self._ac_structure = tp_utils.template_structure_from_gym_space(ac_space)
    self.infserver_addr = infserver_addr
    self.compress = compress  # send compressed data to infserver
    self.n_v = n_v  # number of reward channels

    policy_config = {} if policy_config is None else policy_config
    if 'batch_size' not in policy_config:
      # batch_size is not 1 only when vec_env is used
      policy_config['batch_size'] = 1
    self.batch_size = policy_config['batch_size']
    if 'test' not in policy_config:
      policy_config['test'] = True

    self.nc = policy.net_config_cls(ob_space, ac_space, **policy_config)
    self.rnn = (False if 'use_lstm' not in policy_config
                else policy_config['use_lstm'])
    # numpy rnn state stuff (if any)
    self._last_state = None
    if not self.rnn:
      self._hs_len = None
      self._state = None
      self._start_mask = None
    else:
      self._hs_len = self.nc.hs_len
      if self.batch_size == 1:
        if not hasattr(self.nc, 'reset_hs_func') or (
            self.nc.reset_hs_func is None):
          self._state = np.zeros(shape=(self._hs_len,), dtype=np.float32)
        else:
          # for the case when hs_0 != 0
          self._state = self.nc.reset_hs_func()
        self._start_mask = None
      else:
        # vectorized env
        if not hasattr(self.nc, 'reset_hs_func') or (
            self.nc.reset_hs_func is None):
          self._state = np.zeros(shape=(self.batch_size, self._hs_len),
                                 dtype=np.float32)
        else:
          raise NotImplementedError
        self._start_mask = np.zeros(shape=(self.batch_size,), dtype=np.float32)

    if infserver_addr is None:
      # build the net
      if use_gpu_id < 0:  # not using GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
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

  def set_start_mask(self, start_idx_list):
    assert self.batch_size > 1, 'Non-vec_env should not call this method.'
    self._start_mask = np.zeros(shape=(self.batch_size,))
    self._start_mask[start_idx_list] = True

  def load_model(self, np_params):
    if self.infserver_addr is None:
      self.sess.run(self.params_assign_ops[:len(np_params)], feed_dict={
        ph: np_p for ph, np_p in zip(self.params_ph, np_params)})

  def reset(self, obs=None):
    if self._hs_len is None:
      return
    if self.batch_size == 1:
      if not hasattr(self.nc, 'reset_hs_func') or (
          self.nc.reset_hs_func is None):
        self._state = np.zeros(shape=(self._hs_len,), dtype=np.float32)
      else:
        # for the case when hs_0 != 0
        self._state = self.nc.reset_hs_func()
      self._start_mask = None
    else:
      # vectorized env
      if not hasattr(self.nc, 'reset_hs_func') or (
          self.nc.reset_hs_func is None):
        self._state = np.zeros(shape=(self.batch_size, self._hs_len),
                               dtype=np.float32)
      else:
        raise NotImplementedError
      self._start_mask = np.zeros(shape=(self.batch_size,), dtype=np.float32)

  def _forward(self, obs, fetches=None, action=None):
    if self.infserver_addr is None:
      def _feed_obs(obs, action):
        """prepare feed_dict from input obs"""
        if self.batch_size == 1:
          feed_dict = {ob_ph: [ob_np] for ob_ph, ob_np
                       in
                       zip(nest.flatten(self.inputs_ph.X), nest.flatten(obs))}
          if self._state is not None:
            feed_dict[self.inputs_ph.S] = [self._state]
            # here is one-step lstm inference; hs is fed by last net output
            # or reset by zeros for the start of an episode, so inputs_ph.M
            # which indicates the start of an episode can always be set as 0
            # as the following code line, but in the _push_data_to_learner
            # function, mask should correctly be the start of the episode,
            # since it will be send to learner which does not touch the env
            # but only the data
            feed_dict[self.inputs_ph.M] = [np.zeros(shape=())]
          if action is not None:
            assert self.inputs_ph.A is not None
            for ac_ph, ac_np in zip(nest.flatten(self.inputs_ph.A),
                                    nest.flatten(action)):
              feed_dict[ac_ph] = [ac_np]
        else:
          # This is only for vec_env
          feed_dict = {ob_ph: ob_np for ob_ph, ob_np
                       in
                       zip(nest.flatten(self.inputs_ph.X), nest.flatten(obs))}
          if self._state is not None:
            feed_dict[self.inputs_ph.S] = self._state
            feed_dict[self.inputs_ph.M] = self._start_mask
          if action is not None:
            assert self.inputs_ph.A is not None
            for ac_ph, ac_np in zip(nest.flatten(self.inputs_ph.A),
                                    nest.flatten(action)):
              feed_dict[ac_ph] = ac_np
        return feed_dict

      fetches = {} if fetches is None else fetches
      feed_dict = _feed_obs(obs, action)
      if self._state is not None:
        fetches['state'] = self.net_out.S
      ret = self.sess.run(fetches, feed_dict=feed_dict)
    else:
      data = [obs]
      if action is not None:
        data.append(action)
      if self._state is not None:
        data.extend([self.state, np.array(False)])
      ret = self.apis.request_output(self.ds.structure(data))
    if self.batch_size == 1:
      ret = dict([(k, _squeeze_batch_size_singleton_dim(v))
                  for k, v in ret.items()])
    if self._state is not None:
      self._last_state = self._state
      self._state = ret.pop('state')
    return ret

  def forward_squeezed(self, *args, **kwargs):
    # return action, other params for rl
    raise NotImplementedError

  def step(self, *args, **kwargs):
    # return action
    raise NotImplementedError


class PGAgent(Agent):
  """An agent that carries Policy Gradient compatible policy Neural Network.

  E.g., policy NNs for PPO, VTrace, etc. It only supports
  tpolicies.net_zoo.mnet_v5 or newer Neural Net.
  """

  def forward_squeezed(self, obs):
    """ For rl training usage, return action and other rl variables.

    Self agent will call this method
    """
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
      # other fetches; must be used together with post_process_data in actor
      fetches.update(self.net_out.endpoints)
    else:
      fetches = None
    ret = self._forward(obs, fetches=fetches)
    other_net_out = OrderedDict()
    if self.infserver_addr is None:
      for key in self.net_out.endpoints:
        other_net_out[key] = ret.pop(key)
    ret['state'] = self._last_state
    return ret.pop('a'), ret, other_net_out

  def step(self, obs, argmax=False, endpoints=None):
    """ Only for step, return action only. In PG method, this func is only

    called for opponents inference, because opponents do not return extra
    variables, while self agent returns extra RL variables such as neglogp
    """
    if self.infserver_addr is None:
      # prepare fetches dict
      if not argmax:
        fetches = {
          'a': nest.map_structure_up_to(self._ac_structure,
                                        lambda head: head.sam,
                                        self.net_out.self_fed_heads),
        }
      else:
        fetches = {
          'a': nest.map_structure_up_to(self._ac_structure,
                                        lambda head: head.argmax,
                                        self.net_out.self_fed_heads),
        }
        if endpoints:
          for attr_key in endpoints:
            fetches.update({attr_key: self.net_out.endpoints[attr_key]})
    else:
      fetches = None
    ret = self._forward(obs, fetches=fetches)
    if endpoints:
      return ret.pop('a'), ret
    else:
      return ret['a']

  def update_state(self, obs):
    fetches = {}
    self._forward(obs, fetches=fetches)
    return self._state

  def head_param(self, obs, action=None):
    if self.infserver_addr is None:
      if action is None:
        assert self.net_out.self_fed_heads is not None
        heads = self.net_out.self_fed_heads
      else:
        assert self.net_out.outer_fed_heads is not None
        heads = self.net_out.outer_fed_heads
      fetches = {'flatparam': nest.map_structure_up_to(
        self._ac_structure, lambda head: head.flatparam, heads)}
    else:
      fetches = None
    ret = self._forward(obs, fetches, action)
    return ret['flatparam']


class DDPGAgent(PGAgent):
  def forward_squeezed(self, obs):
    if self.infserver_addr is None:
      # prepare fetches dict
      fetches = {
        'a': nest.map_structure_up_to(self._ac_structure, lambda head: head.sam,
                                      self.net_out.self_fed_heads),
      }
    else:
      fetches = None
    ret = self._forward(obs, fetches=fetches)
    if self._state is not None:
      ret['state'] = self._last_state
    return ret.pop('a'), ret


class DistillAgent(Agent):
  """An agent that carries teacher policy outputs for distillation.
  """

  def forward_squeezed(self, obs):
    if self.infserver_addr is None:
      # prepare fetches dict
      fetches = {
        'a': nest.map_structure_up_to(self._ac_structure,
                                      lambda head: head.sam,
                                      self.net_out.self_fed_heads),
        'flatparam': nest.map_structure_up_to(self._ac_structure,
                                         lambda head: head.flatparam,
                                         self.net_out.self_fed_heads),
      }
    else:
      fetches = None
    ret = self._forward(obs, fetches=fetches)
    ret['state'] = self._last_state
    return ret.pop('a'), ret

  def update_state(self, obs):
    fetches = {}
    self._forward(obs, fetches=fetches)
    return self._state


class GAILAgent(Agent):
  def __init__(self, *args, **kwargs):
    super(GAILAgent, self).__init__(*args, **kwargs)
    self.inputs_ph = self.inputs_ph[0]

  def forward_squeezed(self, obs, action):
    fetches = {
      'reward': self.net_out.reward,
    }
    # return action, other params for rl
    ret = self._forward(obs, fetches=fetches, action=action)
    ret['state'] = self._last_state
    return ret.pop('reward'), ret


class GAILExpertAgent(Agent):
  def forward_squeezed(self, obs):
    if self.infserver_addr is None:
      # prepare fetches dict
      fetches = {
        'a': nest.map_structure_up_to(self._ac_structure, lambda head: head.sam,
                                      self.net_out.self_fed_heads),
      }
    else:
      fetches = None
    ret = self._forward(obs, fetches=fetches)
    ret['state'] = self._last_state
    return ret.pop('a'), ret

  def update_state(self, obs):
    fetches = {}
    self._forward(obs, fetches=fetches)
    return self._state


def _squeeze_batch_size_singleton_dim(st):
  return nest.map_structure(
    lambda x: np.squeeze(x, axis=0) if isinstance(x, np.ndarray) else x, st)


def _insert_batch_size_singleton_dim(st):
  return nest.map_structure(
    lambda x: np.expand_dims(x, axis=0) if isinstance(x, np.ndarray) else x, st)
