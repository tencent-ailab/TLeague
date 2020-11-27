from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
if sys.version_info.major > 2:
  xrange = range

import tensorflow as tf
from tensorflow.contrib.framework import nest
from tleague.learners.pg_learner import PGLearner
from tleague.utils.data_structure import PPOData


class PPOLearner(PGLearner):
  """Learner for PPO."""
  def __init__(self, league_mgr_addr, model_pool_addrs, learner_ports,
               rm_size, batch_size, ob_space, ac_space, policy, gpu_id,
               policy_config={}, ent_coef=1e-2, distill_coef=1e-2,
               vf_coef=0.5, max_grad_norm=0.5, rwd_shape=False,
               pub_interval=500, log_interval=100, save_interval=0,
               total_timesteps=5e7, burn_in_timesteps=0,
               learner_id='', batch_worker_num=4, pull_worker_num=2,
               unroll_length=32, rollout_length=1,
               use_mixed_precision=False, use_sparse_as_dense=True,
               adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-5,
               ep_loss_coef=None, **kwargs):
    super(PPOLearner, self).__init__(
      league_mgr_addr, model_pool_addrs, learner_ports, rm_size, batch_size,
      ob_space, ac_space, policy, gpu_id, policy_config, ent_coef, distill_coef,
      vf_coef, max_grad_norm, rwd_shape, pub_interval, log_interval,
      save_interval, total_timesteps, burn_in_timesteps, learner_id,
      batch_worker_num, pull_worker_num, unroll_length, rollout_length,
      use_mixed_precision,  use_sparse_as_dense, adam_beta1, adam_beta2,
      adam_eps, PPOData, **kwargs)
    self.ep_loss_coef = ep_loss_coef or {}