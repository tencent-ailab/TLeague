from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
if sys.version_info.major > 2:
  xrange = range
import tensorflow as tf

from tleague.learners.pg_learner import PGLearner
from tleague.utils.data_structure import DDPGData


class DDPGLearner(PGLearner):
  """Learner for Vtrace."""
  def __init__(self, league_mgr_addr, model_pool_addrs, learner_ports, rm_size,
               batch_size, ob_space, ac_space, policy, gpu_id, tau=0.001,
               **kwargs):
    self.tau = tau
    super(DDPGLearner, self).__init__(
      league_mgr_addr, model_pool_addrs, learner_ports, rm_size, batch_size,
      ob_space, ac_space, policy, gpu_id, data_type=DDPGData, **kwargs)

  @staticmethod
  def get_target_updates(vars, target_vars, tau):
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
      init_updates.append(tf.assign(target_var, var))
      soft_updates.append(
        tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)


  def _build_ops(self):
    super(DDPGLearner, self)._build_ops()
    self.target_params = tf.trainable_variables(scope='target_model')
    self.target_params_vf = tf.trainable_variables(scope='target_model/vf')
    init_updates, soft_updates = self.get_target_updates(self.params,
                                                         self.target_params,
                                                         self.tau)
    _, vf_soft_updates = self.get_target_updates(self.params_vf,
                                                 self.target_params_vf,
                                                 self.tau)

    def _prepare_td_map(lr, cliprange, lam, weights):
      # for reward shaping weights
      if weights is None:
        assert not self.rwd_shape
        td_map = {self.LR: lr, self.CLIPRANGE: cliprange}
      else:
        td_map = {self.LR: lr, self.CLIPRANGE: cliprange,
                  self.rwd_weights: weights}
      # for lambda of the td-lambda
      if self.LAM is not None:
        td_map[self.LAM] = lam
      return td_map

    def train_batch(lr, cliprange, lam, weights=None):
      td_map = _prepare_td_map(lr, cliprange, lam, weights)
      return self.sess.run(
        self.losses + [self.clip_grad_norm, self.nonclip_grad_norm,
                       self.param_norm, self._train_batch, soft_updates],
        feed_dict=td_map
      )[0:len(self.loss_names)]

    def burn_in(lr, cliprange, lam, weights=None):
      td_map = _prepare_td_map(lr, cliprange, lam, weights)
      return self.sess.run(
        self.losses + [self.clip_grad_norm_vf, self.nonclip_grad_norm_vf,
                       self.param_norm, self._burn_in, vf_soft_updates],
        feed_dict=td_map
      )[0:len(self.loss_names)]

    self.train_batch = train_batch
    self.burn_in = burn_in
    self.sess.run(init_updates)