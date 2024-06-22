from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import time

import numpy as np
from tleague.actors.actor import Actor
from tleague.actors.agent import GAILExpertAgent
from tleague.utils import logger
from tleague.utils.data_structure import DiscAgentData


class GAILExpertActor(Actor):
  """
  GAIL expert actor, similar to pure distillation actor
  All codes for n_agents == 1
  """

  def __init__(self, env, policy, league_mgr_addr, model_pool_addrs,
               expert_model_path,
               **kwargs):
    """ All kwargs are for base Actor.
    Do not touch them or move args in them before. """
    super(GAILExpertActor, self).__init__(env, policy, league_mgr_addr,
                                          model_pool_addrs,
                                          data_type=DiscAgentData,
                                          age_cls=GAILExpertAgent, **kwargs)
    ob_space = self.env.observation_space.spaces[self._learning_agent_id]
    ac_space = self.env.action_space.spaces[self._learning_agent_id]
    self.ds = DiscAgentData(ob_space, ac_space)
    self.expert_model_path = expert_model_path

  def _update_agents_model(self, task):
    """ Fixed self model; never update it in this episode;
    once done the actor will relaunch a new episode; see what's done in Actor
    """
    logger.log('entering _update_agents_model', 'steps: {}'.format(self._steps),
               level=logger.DEBUG + 5)
    if self.self_infserver_addr is None:
      # Only update teacher model once
      if self.expert_model_path is not None and len(self.expert_model_path) > 0:
        model1 = pickle.load(open(self.expert_model_path, 'rb'))
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

  def _push_data_to_learner(self, data_queue):
    """ In PGLearner, the TD-lambda return is computed here. In distill learner,
    there is no need to compute rewards

    :param data_queue:
    :return:
    """
    logger.log('entering _push_data_to_learner steps: {}'.format(self._steps),
               level=logger.DEBUG + 5)
    me_id = self._learning_agent_id  # short name
    last_obs, actions = data_queue.get()
    push_times = 0
    t0 = time.time()
    while True:
      data_model_id = self.task.model_key1
      unroll = []
      while True:
        # extend the unroll until a desired length
        me_action = actions[me_id]
        if isinstance(me_action, list):
          me_action = tuple(me_action)
        data = [last_obs[me_id], me_action]
        data = self.ds.structure(data)
        unroll.append(data)
        last_obs, actions = data_queue.get()
        if len(unroll) >= self._unroll_length:
          break
      # use data server v3 format below
      flatten_unroll = [self.ds.flatten(_data) for _data in unroll]
      shapes = tuple(e.shape for e in flatten_unroll[0])
      unroll_np = np.concatenate(
        [b.reshape(-1) for a in flatten_unroll for b in a])
      self._learner_apis.push_data((data_model_id, unroll_np, {}, shapes))
      logger.log(f"Pushed one unroll to learner at time "
                 f"{time.strftime('%Y%m%d%H%M%S')}",
                 level=logger.DEBUG + 5)
      push_times += 1
      if push_times % 10 == 0:
        push_fps = push_times * self._unroll_length / (time.time() - t0 + 1e-8)
        t0 = time.time()
        logger.log("push fps: {}".format(push_fps))

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
    # agent, env reset
    obs = self.env.reset()
    for agt, ob in zip(self.agents, obs):
      agt.reset(ob)
    self._update_agents_model(self.task)  # for agent Neural Net parameters
    t0 = time.time()
    while True:
      self._steps += 1
      # predictions for each agent
      predictions = self._parallel.run((self._agent_pred, ob, i)
                                       for i, ob in enumerate(obs))
      me_prediction = predictions[me_id]
      action, extra_vars = me_prediction[0], me_prediction[1]
      me_action = action
      actions = [me_action] + predictions[oppo_id:]
      # book-keep obs in previous step
      last_obs = obs
      # agent-env interaction
      obs, _, done, info = self.env.step(actions)

      if self._enable_push:
        data_tuple = (last_obs, tuple(actions))
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
        return None, info
