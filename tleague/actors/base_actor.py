from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid
from abc import ABCMeta, abstractmethod

from tleague.league_mgrs.league_mgr_apis import LeagueMgrAPIs
from tleague.learners.learner_apis import LearnerAPIs
from tleague.model_pools.model_pool_apis import ModelPoolAPIs
from tleague.utils import logger, get_ip_hostname
from tleague.utils.tl_types import MatchResult


class BaseActor(metaclass=ABCMeta):

  def __init__(self, league_mgr_addr, model_pool_addrs, learner_addr=None,
               verbose=0, log_interval_steps=51):
    ip, hostname = get_ip_hostname()
    self._actor_id = hostname + '@' + ip + ':' + str(uuid.uuid1())[:8]
    self._learner_id = None
    self._league_mgr_apis = LeagueMgrAPIs(league_mgr_addr)
    self._model_pool_apis = ModelPoolAPIs(model_pool_addrs)
    if learner_addr:
      self._learner_apis = LearnerAPIs(learner_addr)
      self._learner_id = self._learner_apis.request_learner_id()

    self._log_interval_steps = log_interval_steps
    logger.configure(dir=None, format_strs=['stdout'])
    logger.set_level(verbose)
    self.task = None
    self._steps = 0

  def run(self):
    """Run an infinite loop that rollouts the trajectories for each episode."""
    while True:
      self.task = self._request_task()  # one task for one episode
      outcome, info = self._rollout_an_episode()
      self._finish_task(self.task, outcome, info)

  @abstractmethod
  def _rollout_an_episode(self):
    pass

  def _request_task(self):
    """Request the task for this actor."""
    logger.log('entering _request_task', 'steps: {}'.format(self._steps),
               level=logger.DEBUG + 5)
    task = self._league_mgr_apis.request_actor_task(
      self._actor_id, self._learner_id
    )
    logger.log('leaving _request_task', level=logger.DEBUG + 5)
    return task

  def _finish_task(self, task, outcome, info=None):
    """Do stuff (e.g., send match result) when task finishes."""
    info = info or {}
    logger.log('entering _finish_task', 'steps: {}'.format(self._steps),
               level=logger.DEBUG + 5)
    match_result = MatchResult(task.model_key1, task.model_key2, outcome, info)
    self._league_mgr_apis.notify_actor_task_end(self._actor_id, match_result)
    logger.log('leaving _finish_task', level=logger.DEBUG + 5)
