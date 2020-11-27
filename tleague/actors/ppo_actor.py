from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from copy import deepcopy

import numpy as np

from tleague.actors.pg_actor import PGActor
from tleague.utils import logger
from tleague.utils.io import TensorZipper
from tleague.utils.data_structure import PPOData


class PPOActor(PGActor):
  """Actor for PPO."""
  def __init__(self, env, policy, league_mgr_addr, model_pool_addrs,
               policy_config=None, learner_addr=None, unroll_length=32,
               update_model_freq=32, n_v=1, verbose=0, rwd_shape=True,
               log_interval_steps=51, distillation=False, replay_dir=None,
               agent_cls=None, version='v1', self_infserver_addr=None,
               distill_infserver_addr=None, compress=True):
    super(PPOActor, self).__init__(
      env, policy, league_mgr_addr, model_pool_addrs, policy_config,
      learner_addr, unroll_length, update_model_freq, n_v, verbose, rwd_shape,
      log_interval_steps, distillation, replay_dir, agent_cls, version,
      self_infserver_addr, distill_infserver_addr, compress,
      data_type=PPOData)

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
    if self.use_oppo_obs:
      value, state, neglogpac, oppo_state = other_vars
    else:
      value, state, neglogpac = other_vars
      oppo_state = None

    # loop infinitely to make the unroll on and on
    while True:
      data_model_id = self.task.model_key1
      mb_rewards, mb_values, mb_dones, mb_skips = [], [], [], []
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
          data = [last_obs[me_id], me_action, neglogpac]
          if self.rnn:
            # hidden state and temporal mask for rnn
            data.extend([state, np.array(mask, np.bool)])
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
              data.append(oppo_state)
          data = self.ds.structure(data)
          data.V = value
          data.R = 0.0 # filled later by td_lambda return
          mb_values.append(value)
          mb_rewards.append(reward)
          mb_dones.append(done)
          # Notice: a new episode must starts with a valid obs, not None obs,
          # which is correct currently. Otherwise, mask will be incorrect since
          # it is decided by the last frame's done
          mask = done
          unroll.append(data)
          mb_skips.append(0)
        else:
          mb_skips[-1] += 1
          mb_rewards[-1] += (self._gamma ** mb_skips[-1]) * reward
          mb_dones[-1] += done

        last_obs, actions, reward, info, done, other_vars = data_queue.get()
        if self.use_oppo_obs:
          value, state, neglogpac, oppo_state = other_vars
        else:
          value, state, neglogpac = other_vars
        if done:
          info = deepcopy(info)
          info['outcome'] = self.log_outcome(info)
          infos.append(info)
        if mask and self.distillation:
          self._update_distill_agent_model()
          self.distill_agent.reset(last_obs[me_id])

        if len(unroll) >= self._unroll_length and last_obs[me_id] is not None:
          # need to collect a complete Noop duration
          break

      last_gae_lam = 0
      for t in reversed(range(self._unroll_length)):
        next_values = (value if t == self._unroll_length - 1
                       else mb_values[t + 1])
        delta = (mb_rewards[t] + (self._gamma ** (mb_skips[t]+1))
                 * next_values * (1 - mb_dones[t]) - mb_values[t])
        last_gae_lam = (delta + (self._gamma ** (mb_skips[t]+1))
                        * self._lam * (1 - mb_dones[t]) * last_gae_lam)
        unroll[t].R = np.array(last_gae_lam + mb_values[t], np.float32)
      compressed_unroll = [
        TensorZipper.compress(self.ds.flatten(_data)) for _data in unroll
      ]
      self._learner_apis.push_data((data_model_id, compressed_unroll, infos))
      logger.log(f"Pushed one unroll to learner at time "
                 f"{time.strftime('%Y%m%d%H%M%S')}",
                 level=logger.DEBUG + 5)
