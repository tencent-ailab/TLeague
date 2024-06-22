from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

from tleague.actors.actor import Actor
from tleague.actors.agent import PGAgent
from tleague.utils import logger
from tleague.utils.io import TensorZipper
from tleague.utils.data_structure import PPOData


class PPOActor(Actor):
  """Actor for PPO."""
  def __init__(self, env, policy, league_mgr_addr, model_pool_addrs, data_server_version=None,
               **kwargs):
    super(PPOActor, self).__init__(env, policy, league_mgr_addr,
                                   model_pool_addrs, data_type=PPOData,
                                   age_cls=PGAgent, **kwargs)
    self._data_server_version = data_server_version

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
    push_times = 0
    t0 = time.time()
    while True:
      data_model_id = self.task.model_key1
      mb_rewards, mb_values, mb_dones, mb_skips = [], [], [], []
      unroll = []
      infos = []
      mask = False  # For the first frame in an unroll, there is no need to care
      # about whether it is just a start of a new episode, because even if it is a
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
          data = [last_obs[me_id], me_action, other_vars['neglogp']]
          if self.rnn:
            # hidden state and temporal mask for rnn
            data.extend([other_vars['state'], np.array(mask, np.bool)])
          if self.distillation:
            # teacher logits
            head_param = (self.distill_agent.head_param(last_obs[me_id], me_action)
                          if last_obs[me_id] is not None else None)
            data.append(head_param)
          if self.use_oppo_obs:
            # for fully centralized value net
            data.append(last_obs[oppo_id])
            if self.rnn:
              # oppo hidden state for rnn; mask same as self_agent
              data.append(other_vars['oppo_state'])
          data = self.ds.structure(data)
          data.V = other_vars['v']
          data.R = 0.0 # filled later by td_lambda return
          mb_values.append(other_vars['v'])
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
        if done:
          infos.append(info)
        if mask and self.distillation:
          self._update_distill_agent_model()
          self.distill_agent.reset(last_obs[me_id])

        if len(unroll) >= self._unroll_length and last_obs[me_id] is not None:
          # need to collect a complete Noop duration
          break

      last_gae_lam = 0
      for t in reversed(range(self._unroll_length)):
        next_values = (other_vars['v'] if t == self._unroll_length - 1
                       else mb_values[t + 1])
        delta = (mb_rewards[t] + (self._gamma ** (mb_skips[t]+1))
                 * next_values * (1 - mb_dones[t]) - mb_values[t])
        last_gae_lam = (delta + (self._gamma ** (mb_skips[t]+1))
                        * self._lam * (1 - mb_dones[t]) * last_gae_lam)
        unroll[t].R = np.array(last_gae_lam + mb_values[t], np.float32)
      if self._data_server_version == "v3":
        unroll = [self.ds.flatten(_data) for _data in unroll]
        shapes = tuple(data.shape for data in unroll[0])
        unroll_np = np.concatenate([b.reshape(-1) for a in unroll for b in a])
        self._learner_apis.push_data((data_model_id, unroll_np, infos, shapes))
      else:
        compressed_unroll = [
          TensorZipper.compress(self.ds.flatten(_data)) for _data in unroll
        ]
        self._learner_apis.push_data((data_model_id, compressed_unroll, infos))
      logger.log(f"Pushed one unroll to learner at time "
                 f"{time.strftime('%Y%m%d%H%M%S')}",
                 level=logger.DEBUG + 5)
      push_times += 1
      if push_times % 10 == 0:
        push_fps = push_times * self._unroll_length / (time.time() - t0 + 1e-8)
        t0 = time.time()
        logger.log("push fps: {}".format(push_fps))
