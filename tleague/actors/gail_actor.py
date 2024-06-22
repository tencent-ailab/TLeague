from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
from tleague.actors.actor import Actor
from tleague.actors.agent import PGAgent, GAILAgent
from tleague.learners.learner_apis import LearnerAPIs
from tleague.model_pools.model_pool_apis import ModelPoolAPIs
from tleague.utils import logger
from tleague.utils.data_structure import DiscAgentData
from tleague.utils.data_structure import PPOData


class GAILActor(Actor):
  """Actor for GAIL.
  GAIL Actor inherits Actor, default attributes are defined in class Actor,
  here only some variables related to Gail Actor are defined
  """

  def __init__(self, env, policy, league_mgr_addr, model_pool_addrs,
               discriminator, discriminator_config, gail_learner_addr,
               **kwargs):

    super(GAILActor, self).__init__(env, policy, league_mgr_addr,
                                    model_pool_addrs, data_type=PPOData,
                                    age_cls=PGAgent, **kwargs)
    ob_space = self.env.observation_space.spaces[self._learning_agent_id]
    ac_space = self.env.action_space.spaces[self._learning_agent_id]

    self.ds_agent = DiscAgentData(ob_space, ac_space)
    self.discriminator = GAILAgent(discriminator, ob_space, ac_space,
                                   policy_config=discriminator_config)
    self.disc_model = None
    self._gail_learner_apis = LearnerAPIs(gail_learner_addr)
    self._gail_learner_id = self._gail_learner_apis.request_learner_id()
    self._gail_model_pool_apis = ModelPoolAPIs(model_pool_addrs)
    self.gail_task = None
    self._task_reward_coef = discriminator_config['task_reward_coef']
    self._gail_reward_coef = discriminator_config['gail_reward_coef']

  def _should_update_disc_model(self, model, model_key):
    if model is None or model_key != model.key:
      return True
    elif model.is_freezed():
      return False
    else:
      return self._gail_model_pool_apis.pull_attr(
        'updatetime', model_key) > model.updatetime

  def _update_disc_model(self, task):
    logger.log('entering _update_disc_model', f'steps: {self._steps}',
               level=logger.DEBUG + 5)
    if self._should_update_disc_model(self.disc_model, task.model_key1):
      model = self._gail_model_pool_apis.pull_model(task.model_key1)
      self.discriminator.load_model(model.model)
      self.disc_model = model
    logger.log('leaving _update_disc_model', level=logger.DEBUG + 5)

  def _request_gail_task(self):
    """Request the task for this actor."""
    logger.log('entering _request_gail_task', 'steps: {}'.format(self._steps),
               level=logger.DEBUG + 5)
    print("self._learner_id", self._learner_id)
    print("self._gail_learner_id", self._gail_learner_id)
    task = self._league_mgr_apis.request_actor_task(
      self._actor_id, self._gail_learner_id
    )
    logger.log('leaving _request_gail_task', level=logger.DEBUG + 5)
    return task

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
    self.gail_task = self._request_gail_task()
    logger.log('gail task: {}'.format(str(self.gail_task)))
    # agent, env reset
    obs = self.env.reset()
    for agt, ob in zip(self.agents, obs):
      agt.reset(ob)
    self._update_agents_model(self.task)  # for agent Neural Net parameters
    self._update_disc_model(self.gail_task)

    me_reward_sum = 0.0
    disc_reward_sum = 0.0
    self.time_beg = time.time()
    self._update_hyperparam(self.task)
    t0 = time.time()
    while True:
      self._steps += 1
      # predictions for each agent
      predictions = self._parallel.run((self._agent_pred, ob, i)
                                       for i, ob in enumerate(obs))
      me_prediction = predictions[me_id]
      me_action, extra_vars = me_prediction[0], me_prediction[1]
      actions = [me_action] + predictions[oppo_id:]
      # book-keep obs in previous step
      last_obs = obs

      # agent-env interaction
      obs, reward, done, info = self.env.step(actions)

      me_rwd_scalar = self._reward_shape(reward[me_id])
      me_reward_sum += me_rwd_scalar

      disc_reward, disc_other_vars = self.discriminator.forward_squeezed(
        last_obs[me_id], me_action)
      disc_reward_sum += disc_reward

      if self._enable_push:
        # put the interested data (obs, rwd, act, ... for each agent) into the
        # _data_queue, which is watched in another Thread (_push_data_to_learner
        # ) that the data are dequeued and sent to remote Learner
        rwd_to_push = me_rwd_scalar if self.rwd_shape else reward[me_id]
        rwd_to_push = self._task_reward_coef * np.asarray(rwd_to_push,
                                                          np.float32) + \
                      self._gail_reward_coef * disc_reward

        if rwd_to_push.shape == ():
          rwd_to_push = np.asarray([rwd_to_push], np.float32)
        if done:
          outcome = self.log_outcome(info, reward)
          if not isinstance(info, dict):
            info = {}
          info['outcome'] = outcome
        data_tuple = (
        last_obs, tuple(actions), rwd_to_push, info, done, extra_vars)
        if self._data_queue.full():
          logger.log("Actor's queue is full.", level=logger.WARN)
        self._data_queue.put(data_tuple)
        logger.log('successfully put one tuple.', level=logger.DEBUG)

      if self._steps % self._log_interval_steps == 0:
        logger.log('_rollout_an_episode,', 'steps: {},'.format(self._steps),
                   'data qsize: {}'.format(self._data_queue.qsize()),
                   'rollout fps: {}'.format(
                     self._steps / (time.time() - t0 + 1e-8)))

      if done:
        # an episode ends
        if self._replay_dir:
          self._save_replay()
        self.log_kvs(me_reward_sum, info)
        info.update({'disc_reward': disc_reward_sum / self._steps})
        if 'outcome' not in info:
          info['outcome'] = None
        return info['outcome'], info

      if self._update_model_freq and self._steps % self._update_model_freq == 0:
        # time to update the model for each agent
        self._update_agents_model(self.task)
        self._update_disc_model(self.gail_task)

  def _push_data_to_learner(self, data_queue):
    logger.log('entering _push_data_to_learner',
               'steps: {}'.format(self._steps),
               level=logger.DEBUG + 5)
    me_id = self._learning_agent_id  # short name
    oppo_id = self._oppo_agent_id  # short name

    # initialize
    last_obs, actions, reward, info, done, other_vars = data_queue.get()

    # loop infinitely to make the unroll on and on
    push_times = 0
    t0 = time.time()
    while True:
      data_model_id = self.task.model_key1
      mb_rewards, mb_values, mb_dones, mb_skips = [], [], [], []
      unroll_pg = []
      unroll_gail = []
      infos = []
      mask = False  # For the first frame in an unroll, there is no need to care
      # about whether it is just a start of a new episode, because even if it is
      # new start, hidden state is zero and this is equivalent to mask=True. For
      # other cases, mask must be False. So, just set mask=False here.
      while True:
        if last_obs[me_id] is not None:
          # extend the unroll until a desired length
          me_action = actions[me_id]
          if isinstance(me_action, list):
            me_action = tuple(me_action)
          # Make a `data` for this time step. The `data` is a PGData compatible
          # list, see the PGData definition
          data_pg = [last_obs[me_id], me_action, other_vars['neglogp']]
          data_gail = [last_obs[me_id], me_action]
          if self.rnn:
            # hidden state and temporal mask for rnn
            data_pg.extend([other_vars['state'], np.array(mask, np.bool)])

          data_pg = self.ds.structure(data_pg)
          data_gail = self.ds_agent.structure(data_gail)
          data_pg.V = other_vars['v']
          data_pg.R = 0.0  # filled later by td_lambda return
          mb_values.append(other_vars['v'])
          mb_rewards.append(reward)
          mb_dones.append(done)
          # Notice: a new episode must starts with a valid obs, not None obs,
          # which is correct currently. Otherwise, mask will be incorrect since
          # it is decided by the last frame's done
          mask = done
          unroll_pg.append(data_pg)
          unroll_gail.append(data_gail)
          mb_skips.append(0)
        else:
          mb_skips[-1] += 1
          mb_rewards[-1] += (self._gamma ** mb_skips[-1]) * reward
          mb_dones[-1] += done

        last_obs, actions, reward, info, done, other_vars = data_queue.get()
        if done:
          infos.append(info)

        if len(unroll_pg) >= self._unroll_length and last_obs[
          me_id] is not None:
          # need to collect a complete Noop duration
          break

      last_gae_lam = 0
      for t in reversed(range(self._unroll_length)):
        next_values = (other_vars['v'] if t == self._unroll_length - 1
                       else mb_values[t + 1])
        delta = (mb_rewards[t] + (self._gamma ** (mb_skips[t] + 1))
                 * next_values * (1 - mb_dones[t]) - mb_values[t])
        last_gae_lam = (delta + (self._gamma ** (mb_skips[t] + 1))
                        * self._lam * (1 - mb_dones[t]) * last_gae_lam)
        unroll_pg[t].R = np.array(last_gae_lam + mb_values[t], np.float32)

      unroll_pg = [self.ds.flatten(_data) for _data in unroll_pg]
      shapes = tuple(data.shape for data in unroll_pg[0])
      unroll_np = np.concatenate([b.reshape(-1) for a in unroll_pg for b in a])
      self._learner_apis.push_data((data_model_id, unroll_np, infos, shapes))

      unroll_gail = [self.ds_agent.flatten(_data) for _data in unroll_gail]
      shapes_gail = tuple(data.shape for data in unroll_gail[0])
      unroll_np_gail = np.concatenate(
        [b.reshape(-1) for a in unroll_gail for b in a])
      self._gail_learner_apis.push_data(
        (self.gail_task.model_key1, unroll_np_gail, infos, shapes_gail))

      logger.log(f"Pushed one unroll to learner at time "
                 f"{time.strftime('%Y%m%d%H%M%S')}",
                 level=logger.DEBUG + 5)
      push_times += 1
      if push_times % 10 == 0:
        push_fps = push_times * self._unroll_length / (time.time() - t0 + 1e-8)
        t0 = time.time()
        logger.log("push fps: {}".format(push_fps))
