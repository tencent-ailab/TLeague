from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import warnings
from queue import Queue
from threading import Thread

import numpy as np

from tleague.utils import run_parallel
from tleague.actors.base_actor import BaseActor
from tleague.utils import logger
from tleague.utils.tl_types import is_inherit
from tleague.utils.data_structure import DistillData
from tleague.model_pools.model_pool_msg import ModelPoolErroMsg


def _get_oppo_names(env):
  """ return list of opponent names as [oppo_0, oppo_1, oppo_2,...] """
  n_players = len(env.action_space.spaces)
  names = ['oppo_{}'.format(i) for i in range(n_players - 1)]
  return names


def _get_n_players(env):
  """return number of players(agents)"""
  return len(env.action_space.spaces)


class Actor(BaseActor):
  """Actor that carries two or more PGAgents and sends trajectories to learner.

  Agent 0 is viewed as learning agent, i.e., only the trajectories from
  agents[0] will be pushed to the learner.
  """
  def __init__(self, env,
               policy,
               league_mgr_addr,
               model_pool_addrs,
               age_cls,
               data_type,
               policy_config=None,
               distill_policy=None,
               distill_policy_config=None,
               learner_addr=None,
               unroll_length=32,
               update_model_freq=32,
               n_v=1,
               verbose=0,
               rwd_shape=True,
               log_interval_steps=51,
               distillation=False,
               replay_dir=None,
               self_infserver_addr=None,
               distill_infserver_addr=None,
               compress=True,
               use_oppo_obs=False,
               post_process_data=None,
               post_process_spaces=None,
               **kwargs):
    if len(kwargs) > 0:
      for k in kwargs:
        if data_type == DistillData and k == 'pure_distill_type':
          continue
        warnings.warn('Unused args passed in Actor: {}'.format(k))
    super(Actor, self).__init__(league_mgr_addr,
                                model_pool_addrs,
                                learner_addr,
                                verbose=verbose,
                                log_interval_steps=log_interval_steps)

    self.env = env
    # reset for getting ob/act space later in __init__,
    # empty inter_kwargs is okay
    self.n_agents = _get_n_players(env)
    self.env.reset(inter_kwargs=[{} for i in range(self.n_agents)])
    self._learning_agent_id = 0  # require agents[0] be the learning agent
    self._oppo_agent_id = 1  # require agents[1, 2, ...] be the opponents
    self._enable_push = learner_addr is not None
    self._update_model_freq = update_model_freq
    self._unroll_length = unroll_length
    self._gamma = 0.95
    self._lam = 0.9
    self._type = type
    self._reward_weights = None
    self.self_model = None
    self.oppo_model = None
    self.distill_model = None
    self.self_infserver_addr = self_infserver_addr
    self.distill_infserver_addr = distill_infserver_addr
    self.rwd_shape = rwd_shape
    self.should_log_info = True  # TODO(pengsun): make it an argument
    self.rnn = (False if 'use_lstm' not in policy_config
                else policy_config['use_lstm'])
    self._post_process_data = post_process_data
    self._post_process_spaces = post_process_spaces
    self.use_oppo_obs = use_oppo_obs
    if self.use_oppo_obs:
      assert self.n_agents == 2, 'use_oppo_obs=True only supports n_agents==2'
    ob_space = self.env.observation_space.spaces[self._learning_agent_id]
    ac_space = self.env.action_space.spaces[self._learning_agent_id]
    policy_config = {} if policy_config is None else policy_config
    policy_config['use_loss_type'] = 'none'
    policy_config['use_self_fed_heads'] = True
    # batch_size != 1 only for vec_env
    if 'batch_size' not in policy_config:
      policy_config['batch_size'] = 1

    # Create self agent
    self_agt = age_cls(policy, ob_space, ac_space, n_v=n_v, scope_name="self",
                       policy_config=policy_config, use_gpu_id=-1,
                       infserver_addr=self_infserver_addr, compress=compress)

    # Create other agents; Opponent does not use value heads.
    # NOTE: After removing opponent's value heads, update_model for opponent
    # in actor updates with a full net with more parameters, so the parameters
    # of value heads should be located after the policy (as it currently is in
    # sc2)
    if 'use_value_head' not in policy_config:
      policy_config['use_value_head'] = False
    if 'n_agents' in kwargs and kwargs['n_agents'] == 1:
      self.agents = [self_agt]
    else:
      self.agents = [self_agt] + [
        age_cls(policy, ob_space, ac_space, n_v=n_v, scope_name=scope_name,
                policy_config=policy_config, use_gpu_id=-1, infserver_addr=None)
        for ob_space, ac_space, scope_name in zip(
          self.env.observation_space.spaces[self._oppo_agent_id:],
          self.env.action_space.spaces[self._oppo_agent_id:],
          _get_oppo_names(env)
        )
      ]

    # the data structure
    if data_type == DistillData:
      assert 'pure_distill_type' in kwargs, \
        'DistillActor must be provided with pure_distill_type (ds)'
      self.ds = data_type(ob_space, ac_space, n_v, use_lstm=self.rnn, hs_len=1,
                          distill_type=kwargs['pure_distill_type'])
    else:
      self.ds = data_type(ob_space, ac_space, n_v, use_lstm=self.rnn, hs_len=1,
                          distillation=distillation, use_oppo_obs=use_oppo_obs)
    if self._enable_push:
      # Start a data-sending Thread that watches the _data_queue, see also the
      # self._push_data_to_learner() method
      self._data_queue = Queue(unroll_length)
      self._push_thread = Thread(target=self._push_data_to_learner,
                                 args=(self._data_queue,))
      self._push_thread.daemon = True
      self._push_thread.start()

    # distillation (i.e., the teacher-student KL regularization)
    self.distillation = distillation and self._enable_push
    if self._post_process_spaces:
      # Note: post_process_data is defined after actor agent, before
      # distill_agent. This indicates that actor agent takes original
      # env ob_space and ac_space
      ob_space, ac_space = self._post_process_spaces(ob_space, ac_space)
    if self.distillation:
      # This distillation indicates the type of distillation
      # in AlphaStar and TStarBot-X, which use student to
      # generate data. Standard distillation uses teacher to
      # generate data
      distill_policy_config['use_self_fed_heads'] = False
      distill_policy_func = policy if distill_policy is None else distill_policy
      self.distill_agent = \
        age_cls(distill_policy_func, ob_space, ac_space, n_v=n_v, scope_name="distill",
                policy_config=distill_policy_config, use_gpu_id=-1,
                infserver_addr=distill_infserver_addr, compress=compress)
    self._replay_dir = replay_dir
    self._parallel = run_parallel.RunParallel()

  def _rollout_an_episode(self):
    """Perform trajectory rollout until an episode ends.

    Data are produced by env-agent interaction at each time step. The data are
    put in the _data_queue, and will be sent to (remote) Learner in a separate
    Thread.
    """
    self._steps = 0
    me_id = self._learning_agent_id  # short name
    oppo_id = self._oppo_agent_id  # short name
    logger.log('episode begins with the task: {}'.format(str(self.task)))

    # passing me and oppo hyperparams to the arena interface
    assert self.task.hyperparam is not None
    oppo_hyperparam = None
    if self.n_agents > 1:
      logger.log('pulling oppo hyperparam of model key {}'.format(
        self.task.model_key2))
      oppo_hyperparam = self._model_pool_apis.pull_attr(attr='hyperparam',
                                                        key=self.task.model_key2)
      logger.log('Done pulling oppo hyperparam')
    oppo_inter_kwargs = ({} if oppo_hyperparam is None
                         else oppo_hyperparam.__dict__)
    inter_kwargs = ([self.task.hyperparam.__dict__]
                    + [oppo_inter_kwargs] * (self.n_agents - 1))

    # agent, env reset
    obs = self.env.reset(inter_kwargs=inter_kwargs)
    for agt, ob in zip(self.agents, obs):
      agt.reset(ob)
    self._update_agents_model(self.task)  # for agent Neural Net parameters

    me_reward_sum = 0.0
    reward_sum_vec = None
    self.time_beg = time.time()
    self._update_hyperparam(self.task)
    self._changed_task = False
    t0 = time.time()
    while True:
      self._steps += 1
      # predictions for each agent
      predictions = self._parallel.run((self._agent_pred, ob, i)
                                       for i, ob in enumerate(obs))
      me_prediction = predictions[me_id]
      me_action, extra_vars, other_net_out = me_prediction[0], me_prediction[1], me_prediction[2]
      actions = [me_action] + predictions[oppo_id:]
      # book-keep obs in previous step
      last_obs = obs

      # agent-env interaction
      obs, reward, done, info = self.env.step(actions)

      me_rwd_scalar = self._reward_shape(reward[me_id])
      me_reward_sum += me_rwd_scalar

      if reward_sum_vec is not None:
        reward_sum_vec += np.array(reward)
      else:
        reward_sum_vec = np.array(reward)

      if self._enable_push:
        # put the interested data (obs, rwd, act, ... for each agent) into the
        # _data_queue, which is watched in another Thread (the _push_data_to_learner()
        # method) that the data are dequeued and sent to remote Learner
        rwd_to_push = me_rwd_scalar if self.rwd_shape else reward[me_id]
        rwd_to_push = np.asarray(rwd_to_push, np.float32)
        if rwd_to_push.shape == ():
          rwd_to_push = np.asarray([rwd_to_push], np.float32)
        if self.use_oppo_obs:
          extra_vars['oppo_state'] = self.agents[self._oppo_agent_id]._last_state
        if done:
          # outcome = self.log_outcome(info, reward)
          outcome = self.log_outcome(info, reward_sum_vec)
          if not isinstance(info, dict):
            info = {}
          info['outcome'] = outcome
        data_tuple = (last_obs, tuple(actions), rwd_to_push, info, done, extra_vars)
        if self._post_process_data:
          # if 'post_process_data' not in info:
          #   # the original tleague way for sc2
          #   zipped_data = zip(last_obs, actions)
          # else:
          #   assert len(last_obs) == len(actions) == len(info['post_process_data']), \
          #     'info[post_process_data] must comply with the league training format, ' \
          #     'which matches the agent number dimension.'
          #   zipped_data = zip(last_obs, actions, info.pop('post_process_data'))
          # data_tuple = (*zip(*[self._post_process_data(*x) for x in zipped_data]),
          #               rwd_to_push, info, done, extra_vars)
          data_tuple = self._post_process_data(*data_tuple, other_net_out)
        if self._data_queue.full():
          logger.log("Actor's queue is full.", level=logger.WARN)
        self._data_queue.put(data_tuple)
        logger.log('successfully put one tuple.', level=logger.DEBUG)

      if self._steps % self._log_interval_steps == 0:
        logger.log('_rollout_an_episode,', 'steps: {},'.format(self._steps),
                   'data qsize: {}'.format(self._data_queue.qsize()),
                   'rollout fps: {}'.format(self._steps / (time.time() - t0 + 1e-8)))

      if done:
        # an episode ends
        if self._replay_dir:
          self._save_replay()
        self.log_kvs(me_reward_sum, info)
        if self._changed_task:
          # if the actor task has changed during an episode, then it indicates
          # that the model has changed during the episode and so the info['outcome']
          # should not be counted for that player in league
          return None, info
        else:
          if 'outcome' not in info:
            info['outcome'] = None
          return info['outcome'], info

      if self._update_model_freq and self._steps % self._update_model_freq == 0:
        # time to update the model for each agent
        if (self._enable_push and
            self._model_pool_apis.pull_attr(
                'freezetime', self.task.model_key1) is not None):
          # Current task (learning period) finishes, start a new task or continue
          self._finish_task(self.task, None)  # notify early abort
          last_task = self.task
          self.task = self._request_task()  # try to continue
          if not is_inherit(last_task.model_key1, self.task.model_key1):
            self.log_kvs(me_reward_sum, info)
            return None, info
          if last_task.model_key2 != self.task.model_key2:
            self._changed_task = True
        self._update_agents_model(self.task)

  def _agent_pred(self, ob, i):
    """Feed the observations and do the predictions for each agent."""
    if i == self._learning_agent_id:
      # see what the agent.forward_squeezed() returns
      output = (self.agents[i].forward_squeezed(ob) if ob is not None
                else (None, {}))
    else:
      output = self.agents[i].step(ob) if ob is not None else None
    return output

  def _save_replay(self):
    print('PPOAcotr: trying to save replay to {}'.format(self._replay_dir))
    # NOTE: this is an extremely dirty hacking
    try:
      self.env.env.unwrapped.env.save_replay(self._replay_dir)
    except Exception as e:
      print('error when saving replay: {}'.format(e))
      pass

  def _should_update_model(self, model, model_key):
    if model is None:
      return True
    elif isinstance(model, ModelPoolErroMsg):
      return True
    elif model_key != model.key:
      return True
    elif model.is_freezed():
      return False
    else:
      return self._model_pool_apis.pull_attr('updatetime', model_key) > model.updatetime

  def _update_agents_model(self, task):
    """Update the model (i.e., Neural Net parameters) for each agent.

    The learning agent uses model1, all the other opponent(s) use model2 """
    logger.log('entering _update_agents_model', 'steps: {}'.format(self._steps),
               level=logger.DEBUG + 5)
    if (self.self_infserver_addr is None
        and self._should_update_model(self.self_model, task.model_key1)):
      model1 = self._model_pool_apis.pull_model(task.model_key1)
      me_id = self._learning_agent_id  # short name
      self.agents[me_id].load_model(model1.model)
      self.self_model = model1
    if self.n_agents > 1 and self._should_update_model(self.oppo_model,
                                                       task.model_key2):
      model2 = self._model_pool_apis.pull_model(task.model_key2)
      oppo_id = self._oppo_agent_id  # short name
      for agt in self.agents[oppo_id:]:
        agt.load_model(model2.model)
      self.oppo_model = model2
    logger.log('leaving _update_agents_model', level=logger.DEBUG + 5)

  def _update_distill_agent_model(self):
    if self.distill_infserver_addr is not None:
      return
    logger.log('entering _update_distill agent_model', f'steps: {self._steps}',
               level=logger.DEBUG + 5)
    model_key = self.task.hyperparam.distill_model_key
    if self._should_update_model(self.distill_model, model_key):
      model3 = self._model_pool_apis.pull_model(model_key)
      self.distill_agent.load_model(model3.model)
      self.distill_model = model3
    logger.log('leaving _update_distill_agent_model', level=logger.DEBUG + 5)

  def _update_hyperparam(self, task):
    logger.log('entering _update_hyperparam', f'steps: {self._steps}',
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
    assert (len(reward.shape) == 1
            and reward.shape[0] == self._reward_weights.shape[-1]), \
      'reward {}, weights {}'.format(str(reward),
                                     str(self._reward_weights))
    return self._reward_weights.dot(reward)

  def log_kvs(self, reward_sum, info):
    time_end = time.time()
    temp = {
      'producing_fps': self._steps / (time_end - self.time_beg),
      'reward_sum': reward_sum,
      'episode_steps': self._steps,
    }
    logger.logkvs(temp)
    if self.should_log_info:  # log additional info fields
      if isinstance(info, dict):
        logger.logkvs(info)
      else:
        logger.log(info)
    if isinstance(info, dict):
      for k, v in temp.items():
        if k not in info:
          info[k] = v
    logger.dumpkvs()

  def log_outcome(self, info, rwd):
    # Previously this log_outcome function only works for sparse reward game
    # it only records the terminate immediate reward (first channel, win-loss
    # in sc2)
    if not isinstance(info, dict) or 'outcome' not in info:
      logger.log("info['outcome'] not available, get it from reward",
                 level=logger.WARN)
      if self.n_agents > 1:
        rwd_me = rwd[self._learning_agent_id]
        rwd_oppo = rwd[self._oppo_agent_id:]
        try:
          logger.log(f"reward[0] is vector of {len(rwd_me)},"
                     f" get first channel as outcome", level=logger.WARN)
          win_num = sum([r[0] < rwd_me[0] for r in rwd_oppo])
          lose_num = sum([r[0] > rwd_me[0] for r in rwd_oppo])
        except:
          win_num = sum([r < rwd_me for r in rwd_oppo])
          lose_num = sum([r > rwd_me for r in rwd_oppo])
        me_outcome = 1 if win_num > lose_num else -1 if win_num < lose_num else 0
      else:
        me_outcome = None
    else:
      me_outcome = info['outcome'][self._learning_agent_id]
    return me_outcome

  def _push_data_to_learner(self, data_queue):
    """Organize the trajectory unroll data and push the data to a remote learner
     for the learning agent (id 0).

     Invoked in a Thread. Override in derived class."""
    raise NotImplementedError
