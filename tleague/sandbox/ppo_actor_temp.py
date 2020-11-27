from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from queue import Queue
from threading import Thread

import numpy as np
import tensorflow as tf
from gym import spaces
from tleague.actors.base_actor import BaseActor
from tleague.utils import logger
from tleague.utils.tl_types import is_inherit


class PPOAgent(object):
  """ An agent that carries PPO compatible policy Neural Network.  """
  def __init__(self, policy, ob_space, ac_space, n_v=1,
               action_mask=True, scope_name="model"):
    sess = tf.Session()
    self.ob_space = ob_space
    with sess.as_default():
      self.policy = policy(ob_space, ac_space, n_v=n_v, nbatch=1,
                           action_mask=action_mask, scope_name=scope_name)
    self._state = self.policy.initial_state
    params = tf.trainable_variables(scope=scope_name)
    new_params = [tf.placeholder(p.dtype, shape=p.get_shape()) for p in params]
    param_assign_ops = [p.assign(new_p) for p, new_p in zip(params, new_params)]
    tf.global_variables_initializer().run(session=sess)

    def load_model(loaded_params):
      sess.run(param_assign_ops,
               feed_dict={p : v for p, v in zip(new_params, loaded_params)})
    self.load_model = load_model

  def reset(self, obs=None):
    self._state = self.policy.initial_state  # used to store rnn hidden layer

  def act(self, obs):
    if isinstance(self.ob_space, spaces.Tuple):
      feed_obs = [[ob] for ob in obs]
    else:
      feed_obs = [obs]
    if self._state is not None:
      a, v, state, p = self.policy.step(feed_obs, self._state)
      self._last_state = self._state
      self._state = state
    else:
      a, v, state, p = self.policy.step(feed_obs)
      self._last_state = self._state
    if isinstance(a, list):  # action from spaces.Tuple
      return [b.squeeze() for b in a], v.squeeze(), self._last_state, p.squeeze()
    else:
      return a.squeeze(), v.squeeze(), self._last_state, p.squeeze()

  def step(self, obs):
    a, v, state, p = self.act(obs)
    return a


def _get_oppo_names(env):
  """ return list of opponent names as oppo, oppo1, oppo2,... """
  n_players = len(env.action_space.spaces)
  assert n_players > 1, "only one agent, no opponent at all!"
  names = ['oppo']
  if n_players > 2:
    names += ['oppo{}'.format(i + 1) for i in range(n_players - 2)]
  return names


class PPOActor(BaseActor):
  """ Actor that carries two or more PPOAgents. Agent 0 viewed as learning
  agent, i.e., only the trajectories from agents[0] will be pushed to the
  learner. """
  def __init__(self, env, policy, league_mgr_addr, model_pool_addrs,
               learner_addr=None, unroll_length=32, update_model_freq=32,
               n_v=1, verbose=0, rwd_shape=True, log_interval_steps=51):
    super(PPOActor, self).__init__(league_mgr_addr,
                                   model_pool_addrs,
                                   learner_addr,
                                   verbose=verbose,
                                   log_interval_steps=log_interval_steps)
    logger.configure(dir=None, format_strs=['stdout'])
    logger.set_level(verbose)

    self.env = env
    self.env.reset()
    sp = self.env.observation_space.spaces[0]
    sp = spaces.Box(low=0, high=1, shape=sp.shape)
    self.obs_space = spaces.Tuple([sp] * 2)
    self.agents = [
      PPOAgent(policy, ob_space, ac_space, n_v=n_v, scope_name=scope_name)
      for ob_space, ac_space, scope_name in zip(
        self.env.observation_space.spaces, self.env.action_space.spaces,
        ["self"] + _get_oppo_names(env)
      )
    ]
    self.env.close()
    self._learning_agent_id = 0
    self._enable_push = learner_addr is not None
    self._update_model_freq = update_model_freq
    self._unroll_length = unroll_length
    self._gamma = 0.95
    self._lam = 0.9
    self._reward_weights = None
    self.n_v = n_v # reward/value length
    self.models = [None, None]
    self.rwd_shape = rwd_shape
    self.should_log_info = True  # TODO(pengsun): make it an argument
    if self._enable_push:
      self._data_queue = Queue(unroll_length)
      self._push_thread = Thread(target=self._push_data, args=(self._data_queue,))
      self._push_thread.daemon = True
      self._push_thread.start()

  def _rollout_an_episode(self):
    """ perform roullout until one episode done. Data are put in _data_queue,
     which will be sent to remote in a separate thread """
    self._steps = 0

    self.task = self._request_task()
    logger.log('episode begins, task: {}'.format(str(self.task)))

    #obs = self.env.reset()
    obs = self.obs_space.sample()
    for agt, ob in zip(self.agents, obs):
      agt.reset(ob)
    self._update_agents_model(self.task)

    me_id = self._learning_agent_id  # short name
    reward_sum = 0.0
    time_beg = time.time()
    self._update_hyperparam(self.task)
    while True:
      self._steps += 1
      output = self.agents[me_id].act(obs[me_id])
      action, other_vars = output[0], output[1:]
      oppo_actions = [agt.step(ob) for agt, ob in
                      zip(self.agents[me_id + 1:], obs[me_id + 1:])]
      last_obs = obs
      obs = self.obs_space.sample()
      reward = [np.zeros(shape=(17,)), np.zeros(shape=(17,))]
      done = self._steps == 300
      info = {'outcome': [0, 0]}
      #obs, reward, done, info = self.env.step([action] + oppo_actions)

      rwd = self._reward_shape(reward[me_id])
      reward_sum += rwd
      if self._enable_push:
        if self._data_queue.full():
          logger.log("Actor's queue is full.", level=logger.WARN)
        rwd_to_push = rwd if self.rwd_shape else reward[me_id]
        self._data_queue.put((last_obs, tuple([action] + oppo_actions),
                              rwd_to_push, info, done, other_vars))
        logger.log('successfully put one tuple.', level=logger.DEBUG)

      if self._steps % self._log_interval_steps == 0:
        logger.log('_rollout_an_episode,', 'steps: {},'.format(self._steps),
                   'data qsize: {}'.format(self._data_queue.qsize()))

      if done:
        time_end = time.time()
        logger.logkvs({
          'producing_fps': self._steps / (time_end - time_beg),
          'reward_sum': reward_sum,
          'episode_steps': self._steps,
        })
        if self.should_log_info:  # log additional info fields
          logger.logkvs(info)
        logger.dumpkvs()
        if 'outcome' not in info:
          me_outcome = -95678
          logger.log("info['outcome'] not available",
                     'return an arbitrary value', me_outcome, level=logger.WARN)
        else:
          me_outcome = info['outcome'][me_id]
        return me_outcome

      if self._update_model_freq and self._steps % self._update_model_freq == 0:
        if (self._enable_push and
            self._remote.pull_model_attr('freezetime', self.task.model_key1)
            is not None):
          # Current task (learning period) finishes, start a new task or continue
          self._finish_task(self.task, None)  # notify early abort
          last_task = self.task
          self.task = self._request_task()  # try to continue
          if not is_inherit(last_task.model_key1, self.task.model_key1):
            time_end = time.time()
            logger.logkvs({
              'producing_fps': self._steps / (time_end - time_beg),
              'reward_sum': reward_sum,
              'episode_steps': self._steps,
            })
            if self.should_log_info:  # log additional info fields
              logger.logkvs(info)
            logger.dumpkvs()
            return None
        self._update_agents_model(self.task)

  def _update_agents_model(self, task):
    """ learning agent uses model1, all the other opponent(s) use model2 """
    logger.log('entering _update_agents_model', 'steps: {}'.format(self._steps),
               level=logger.DEBUG + 5)
    if (self.models[0] is None or task.model_key1 != self.models[0].key
        or (not self.models[0].is_freezed()
            and self._remote.pull_model_attr('updatetime', task.model_key1) >
            self.models[0].updatetime)):
      model1 = self._remote.pull_model(task.model_key1)
      me_id = self._learning_agent_id
      self.agents[me_id].load_model(model1.model)
      self.models[0] = model1
    if (self.models[1] is None or task.model_key2 != self.models[1].key
        or (not self.models[1].is_freezed()
            and self._remote.pull_model_attr('updatetime', task.model_key2) >
            self.models[1].updatetime)):
      model2 = self._remote.pull_model(task.model_key2)
      for agt in self.agents[1:]:
        agt.load_model(model2.model)
      self.models[1] = model2
    logger.log('leaving _update_agents_model', level=logger.DEBUG + 5)

  def _update_hyperparam(self, task):
    logger.log('entering _update_hyperparam', 'steps: {}'.format(self._steps),
               level=logger.DEBUG + 5)
    if self._enable_push:
      if hasattr(task.hyperparam, 'gamma'):
        self._gamma = task.hyperparam.gamma
      if hasattr(task.hyperparam, 'lam'):
        self._lam = task.hyperparam.lam
      if hasattr(task.hyperparam, 'reward_weights'):
        self._reward_weights = np.array(task.hyperparam.reward_weights,
                                        dtype=np.float32)
    logger.log('leaving _update_hyperparam', level=logger.DEBUG + 5)

  def _reward_shape(self, reward):
    if self._reward_weights is None:
      return reward
    if not isinstance(reward, np.ndarray):
      reward = np.array(reward)
    if len(reward.shape) == 0:
      reward = np.expand_dims(reward, 0)
    assert len(reward.shape) == 1 and reward.shape[0] == self._reward_weights.shape[-1]
    return self._reward_weights.dot(reward)

  def _push_data(self, data_queue):
    """ push trajectory for the learning agent (id 0). Invoked in a thread """
    logger.log('entering _push_data_to_learner', 'steps: {}'.format(self._steps),
               level=logger.DEBUG + 5)
    me_id = self._learning_agent_id  # short name
    last_obs, actions, reward, info, done, other_vars = data_queue.get()
    value, state, neglogpac = other_vars
    while True:
      data_model_id = self.task.model_key1
      mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = (
        [], [], [], [], [], [])
      mb_states = []
      for _ in range(self._unroll_length):
        mb_obs.append(transform_tuple(last_obs[me_id], lambda x: x.copy()))
        mb_actions.append(actions[me_id])
        mb_rewards.append(reward)
        mb_dones.append(done)
        mb_values.append(value)
        mb_neglogpacs.append(neglogpac)
        mb_states.append(state)
        last_obs, actions, reward, info, done, other_vars = data_queue.get()
        value, state, neglogpac = other_vars
      if (isinstance(last_obs[me_id], tuple) or
          isinstance(last_obs[me_id], list)):
        mb_obs = tuple(np.asarray(obs, dtype=obs[me_id].dtype)
                       for obs in zip(*mb_obs))
      else:
        mb_obs = np.asarray(mb_obs, dtype=last_obs[me_id].dtype)
      mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
      if isinstance(actions[me_id], list) or isinstance(actions[me_id], tuple):
        # actions can a list (e.g., from a transformer network)
        mb_actions = tuple(np.squeeze(np.asarray(a, dtype=np.float32))
                           for a in zip(*mb_actions))
      else:
        mb_actions = np.asarray(mb_actions)
      mb_values = np.asarray(mb_values, dtype=np.float32)
      mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
      mb_dones = np.asarray(mb_dones, dtype=np.bool)
      mb_states = np.asarray(mb_states)
      mb_returns = np.zeros_like(mb_rewards)
      mb_advs = np.zeros_like(mb_rewards)
      last_gae_lam = 0
      for t in reversed(range(self._unroll_length)):
        next_values = (value if t == self._unroll_length - 1
                       else mb_values[t + 1])
        delta = (mb_rewards[t] + self._gamma * next_values * (1 - mb_dones[t])
                 - mb_values[t])
        mb_advs[t] = last_gae_lam = (delta + self._gamma * self._lam *
                                     (1 - mb_dones[t]) * last_gae_lam)
      mb_returns = mb_advs + mb_values
      # All done, send them to remote
      self._remote.push_data((data_model_id, mb_obs, mb_returns, mb_dones,
                              mb_actions, mb_values, mb_neglogpacs, mb_states))


class PPOActorFixedOppo(PPOActor):
  """ Actor that carries one PPOAgent and one or more fixed agents as
  opponent(s). Agent 0 viewed as learning agent. """
  def __init__(self, env, policy, oppo_agent_cls, league_mgr_addr, model_pool_addrs,
               unroll_length=32, update_model_freq=32, checkpoint_path=None):
    super(PPOActor, self).__init__(league_mgr_addr,
                                   model_pool_addrs,
                                   learner_addr=None)
    self.env = env
    self.env.reset()
    me_id = self._learning_agent_id  # short name
    self.agents = [PPOAgent(
      policy,
      self.env.observation_space.spaces[me_id],
      self.env.action_space.spaces[me_id],
      scope_name='self'
    )]
    self.agents += [oppo_agent_cls(ac_space)
                    for ac_space in env.action_space.spaces[me_id + 1:]]
    self._enable_push = None
    self._update_model_freq = update_model_freq
    self._unroll_length = unroll_length
    self._gamma = 0.95
    self._lam = 0.9
    self._reward_weights = None
    self.models = [None, None]
    self._checkpoint_path = checkpoint_path

  def _update_agents_model(self, task):
    """ learning agent uses model1, all the other opponent(s) are fixed and
     need no model updating logic. """
    if self._checkpoint_path is not None:
      # TODO(pengsun): confirm why we have the loading model logic here?
      import pickle
      with open(self._checkpoint_path, 'rb') as f:
        print('load model from checkpoint path.')
        model1 = pickle.load(f)
      self.models[0] = model1
    else:
      if (self.models[0] is None or task.model_key1 != self.models[0].key
          or (not self.models[0].is_freezed()
              and self._remote.pull_model_attr('updatetime', task.model_key1) >
              self.models[0].updatetime)):
        model1 = self._remote.pull_model(task.model_key1)
        me_id = self._learning_agent_id
        self.agents[me_id].load_model(model1.model)
        self.models[0] = model1

  def run(self):
    while True:
      outcome = self._rollout_an_episode()
      # self._finish_task(self.task, outcome)


def transform_tuple(x, transformer):
  if isinstance(x, tuple):
    return tuple(transformer(a) for a in x)
  else:
    return transformer(x)
