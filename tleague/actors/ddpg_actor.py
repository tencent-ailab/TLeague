from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np

from tleague.actors.vtrace_actor import VtraceActor
from tleague.actors.agent import DDPGAgent
from tleague.utils import logger
from tleague.utils.io import TensorZipper
from tleague.utils.data_structure import DDPGData


class DDPGActor(VtraceActor):
  """Actor for DDPG."""
  def __init__(self, env, policy, league_mgr_addr, model_pool_addrs, **kwargs):
    super(VtraceActor, self).__init__(env, policy, league_mgr_addr,
                                      model_pool_addrs, data_type=DDPGData,
                                      age_cls=DDPGAgent, **kwargs)

  def _push_data_to_learner(self, data_queue):
    logger.log('entering _push_data_to_learner',
               'steps: {}'.format(self._steps),
               level=logger.DEBUG + 5)
    me_id = self._learning_agent_id  # short name
    oppo_id = self._oppo_agent_id  # short name

    # initialize
    last_obs, actions, reward, info, done, other_vars = data_queue.get()
    if self.distillation:
      self._update_distill_agent_model()
      self.distill_agent.reset(last_obs[me_id])

    # loop infinitely to make the unroll on and on
    while True:
      data_model_id = self.task.model_key1
      mb_skips = []
      unroll = []
      infos = []
      mask = False
      while True:
        if last_obs[me_id] is not None:
          # extend the unroll until a desired length
          me_action = actions[me_id]
          if isinstance(me_action, list):
            me_action = tuple(me_action)
          # Make a `data` for this time step. The `data` is a PGData compatible
          # list, see the PGData definition
          data = [last_obs[me_id], me_action]
          if self.rnn:
            # hidden state and temporal mask for rnn
            data.extend([other_vars['state'], np.array(mask, np.bool)])
          if self.distillation:
            # teacher logits
            logits = (self.distill_agent.logits(last_obs[me_id], me_action)
                      if last_obs[me_id] is not None else None)
            data.append(logits)
          if self.use_oppo_obs:
            # for fully centralized value net
            data.append(last_obs[oppo_id])
            if self.rnn:
              # oppo hidden state for rnn; mask same as self_agent
              data.append(other_vars['oppo_state'])
          data = self.ds.structure(data)
          data.r = reward
          data.discount = 1.0
          # Notice: a new episode must starts with a valid obs, not None obs,
          # which is correct currently. Otherwise, mask will be incorrect since
          # it is decided by the last frame's done
          mask = done
          unroll.append(data)
          mb_skips.append(0)
        else:
          mb_skips[-1] += 1
          # correct cumulated reward and discount factor
          data.r += (self._gamma ** mb_skips[-1]) * reward
          data.discount *= (1 - done) * self._gamma

        last_obs, actions, reward, info, done, other_vars = data_queue.get()
        if done:
          infos.append(info)
        if mask and self.distillation:
          self._update_distill_agent_model()
          self.distill_agent.reset(last_obs[me_id])

        if len(unroll) >= self._unroll_length and last_obs[me_id] is not None:
          # need to collect a complete Noop duration
          break

      compressed_unroll = [
        TensorZipper.compress(self.ds.flatten(_data)) for _data in unroll
      ]
      self._learner_apis.push_data((data_model_id, compressed_unroll, infos))
      logger.log(f"Pushed one unroll to learner at time "
                 f"{time.strftime('%Y%m%d%H%M%S')}",
                 level=logger.DEBUG + 5)
