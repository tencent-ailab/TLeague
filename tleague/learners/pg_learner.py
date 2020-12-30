from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import time

if sys.version_info.major > 2:
  xrange = range

import joblib
import numpy as np
import tensorflow as tf
try:
  import horovod.tensorflow as hvd
  has_hvd = True
except:
  has_hvd = False

from tensorflow.contrib.framework import nest
from tleague.learners.base_learner import BaseLearner
from tleague.learners.data_server import DataServer
from tleague.utils import logger
from tleague.utils.data_structure import PGData


def as_func(obj):
  if isinstance(obj, float):
    return lambda x: obj
  else:
    assert callable(obj)
    return obj


class PGLearner(BaseLearner):
  """Base learner class for Policy Gradient series."""
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
               data_type=PGData, data_server_version='v1',
               decode=False, log_infos_interval=20, ep_loss_coef=None,
               **kwargs):
    super(PGLearner, self).__init__(league_mgr_addr, model_pool_addrs,
                                    learner_ports, learner_id)

    self.LR = tf.placeholder(tf.float32, [])
    """Learning Rate"""

    self.CLIPRANGE = tf.placeholder(tf.float32, [])
    """Learning Rate Clip Range"""

    self.ep_loss_coef = ep_loss_coef or {}
    """Coefficients for those losses from the endpoints."""

    # TODO(pengsun): fix the policy_config default value
    self._init_const(total_timesteps, burn_in_timesteps, batch_size,
                     unroll_length, rwd_shape, ent_coef, vf_coef,
                     pub_interval, log_interval, save_interval,
                     policy, distill_coef, policy_config, rollout_length)

    # allow_soft_placement=True can fix issue when some op cannot be defined on
    # GPUs for tf-1.8.0; tf-1.13.1 does not have this issue
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(gpu_id)
    self.sess = tf.Session(config=config)
    self.use_hvd = has_hvd and hvd.size() > 1
    self.rank = hvd.rank() if self.use_hvd else 0

    # Prepare dataset
    ds = data_type(ob_space, ac_space, self.n_v, use_lstm=self.rnn,
                   hs_len=self.hs_len, distillation=self.distillation)
    self._data_server = DataServer(self._pull_data, rm_size,
                                   unroll_length, batch_size, ds,
                                   gpu_id_list=(0,),
                                   batch_worker_num=batch_worker_num,
                                   pull_worker_num=pull_worker_num,
                                   rollout_length=rollout_length,
                                   prefetch_buffer_size=2,
                                   version=data_server_version,
                                   decode=decode,
                                   log_infos_interval=log_infos_interval)

    # prepare net config
    net_config = policy.net_config_cls(ob_space, ac_space, **policy_config)
    net_config.clip_range = self.CLIPRANGE
    if rwd_shape:
      # make net_config.reward-shaping-weights a tf.placeholder so as to change
      # it during training.
      # NOTE: Assume there is reward_weights_shape in net_config
      # TODO(pengsun): use NetInputsData instead of this quick-and-dirty hacking?
      reward_weights_shape = net_config.reward_weights_shape
      self.rwd_weights = tf.placeholder(tf.float32, reward_weights_shape)
      net_config.reward_weights = self.rwd_weights
    if hasattr(net_config, 'lam'):
      # make net_config.lambda-for-td-lambda a tf.placeholder so as to change it
      #  during training.
      # TODO(pengsun): use NetInputsData instead of this quick-and-dirty hacking?
      self.LAM = tf.placeholder(tf.float32, [])
      net_config.lam = self.LAM
    else:
      self.LAM = None

    # build the policy net
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as model_scope:
      pass
    def create_policy(inputs, nc):
      return policy.net_build_fun(inputs=inputs, nc=nc, scope=model_scope)

    device = '/gpu:{}'.format(0)
    with tf.device(device):
      input_data = self._data_server.input_datas[0]
      if 'use_xla' in policy_config and policy_config['use_xla']:
        try:
          # Use tensorflow's accerlated linear algebra compile method
          with tf.xla.experimental.jit_scope(True):
            model = create_policy(input_data, net_config)
        except:
          logger.log("WARNING: using tf.xla requires tf version>=1.15.")
          model = create_policy(input_data, net_config)
      else:
        model = create_policy(input_data, net_config)
      loss, vf_loss, losses = self.build_loss(model, input_data)
    if self.use_hvd:
      self.losses = [hvd.allreduce(loss) for loss in losses]
    else:
      self.losses = list(losses)
    self.params = tf.trainable_variables(scope='model')
    self.params_vf = tf.trainable_variables(scope='model/vf')
    self.param_norm = tf.global_norm(self.params)

    self.trainer = tf.train.AdamOptimizer(learning_rate=self.LR,
                                          beta1=adam_beta1,
                                          beta2=adam_beta2,
                                          epsilon=adam_eps)
    self.burn_in_trainer = tf.train.AdamOptimizer(
      learning_rate=self.LR,
      epsilon=1e-5
    )  # same as default and IL
    if use_mixed_precision:
      try:
        self.trainer = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(self.trainer)
        self.burn_in_trainer = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(self.burn_in_trainer)
      except:
        logger.warn("using tf mixed_precision requires tf version>=1.15.")
    if self.use_hvd:
      self.trainer = hvd.DistributedOptimizer(
        self.trainer, sparse_as_dense=use_sparse_as_dense)
      self.burn_in_trainer = hvd.DistributedOptimizer(
        self.burn_in_trainer, sparse_as_dense=use_sparse_as_dense)
    grads_and_vars = self.trainer.compute_gradients(loss, self.params)
    grads_and_vars_vf = self.burn_in_trainer.compute_gradients(vf_loss,
                                                               self.params_vf)
    if 'use_lstm' in policy_config and policy_config['use_lstm']:                                                          
      clip_vars = model.vars.lstm_vars
    else:
      clip_vars = []
    grads_and_vars, self.clip_grad_norm, self.nonclip_grad_norm = self.clip_grads_vars(
      grads_and_vars, clip_vars, max_grad_norm)
    grads_and_vars_vf, self.clip_grad_norm_vf, self.nonclip_grad_norm_vf = self.clip_grads_vars(
      grads_and_vars_vf, clip_vars, max_grad_norm)

    self._train_batch = self.trainer.apply_gradients(grads_and_vars)
    self._burn_in = self.burn_in_trainer.apply_gradients(grads_and_vars_vf)
    self.loss_endpoints_names = model.loss.loss_endpoints.keys()
    if self.use_hvd:
      barrier_op = hvd.allreduce(tf.Variable(0.))
      broadcast_op = hvd.broadcast_global_variables(0)
    tf.global_variables_initializer().run(session=self.sess)
    self._build_ops()
    self.sess.graph.finalize()

    self.barrier = lambda: self.sess.run(barrier_op) if self.use_hvd else None
    self.broadcast = lambda: self.sess.run(broadcast_op) if self.use_hvd else None
    self.broadcast()
    # logging stuff
    format_strs = (['stdout', 'log', 'tensorboard', 'csv'] if self.rank == 0
                   else ['stdout', 'log', 'tensorboard', 'csv'])
    logger.configure(
      dir='training_log/{}rank{}'.format(self._learner_id, self.rank),
      format_strs=format_strs
    )

  def run(self):
    logger.log('HvdPPOLearner: entering run()')
    first_lp = True
    while True:
      if self.rank == 0:
        if first_lp:
          first_lp = False
          task = self._query_task()
          if task is not None:
            self.task = task
          else:
            self.task = self._request_task()
        else:
          self.task = self._request_task()
        logger.log('rank{}: done _request_task'.format(self.rank))
      self.barrier()
      if self.rank == 0:
        self._init_task()
        self._notify_task_begin(self.task)
        logger.log('rank{}: done init_task and notify_task_begin...'.format(
          self.rank))
      else:
        self.task = self._query_task()
        logger.log('rank{}: done _query_task...'.format(self.rank))
      self.barrier()

      logger.log('rank{}: broadcasting...'.format(self.rank))
      self.broadcast()
      logger.log('rank{}: done broadcasting'.format(self.rank))
      self._train()
      self._lrn_period_count += 1

  def _train(self, **kwargs):
    self._data_server._update_model_id(self.model_key)
    # Use different model, clear the replay memory
    if (self.last_model_key is None
        or self.last_model_key != self.task.parent_model_key):
      self._data_server.reset()
      if self._lrn_period_count == 0:
        self._need_burn_in = self.burn_in_timesteps > 0
      else:
        self._need_burn_in = True
    else:
      self._need_burn_in = False

    self.barrier()
    nbatch = self.batch_size * hvd.size() if self.use_hvd else self.batch_size
    self.should_push_model = (self.rank == 0)
    self._run_train_loop(nbatch)


  def _init_const(self, total_timesteps, burn_in_timesteps, batch_size,
                  unroll_length, rwd_shape, ent_coef, vf_coef,
                  pub_interval, log_interval, save_interval,
                  policy, distill_coef, policy_config, rollout_length):
    self.total_timesteps = total_timesteps
    self.burn_in_timesteps = burn_in_timesteps
    self._train_batch = []
    self._burn_in = []
    self.batch_size = batch_size
    self.unroll_length = unroll_length
    self.rwd_shape = rwd_shape
    self.ent_coef = ent_coef
    self.vf_coef = vf_coef
    self.pub_interval = pub_interval
    self.log_interval = log_interval
    self.save_interval = save_interval
    self.policy = policy
    self.distillation = (distill_coef != 0)
    self.distill_coef = distill_coef
    self.rnn = (False if 'use_lstm' not in policy_config
                else policy_config['use_lstm'])
    self.hs_len = None
    self.n_v = policy_config['n_v']
    policy_config['batch_size'] = batch_size
    policy_config['rollout_len'] = rollout_length
    if self.rnn:
      self.hs_len = policy_config['hs_len']

  def _build_ops(self):
    ## other useful operators
    self.new_params = [tf.placeholder(p.dtype, shape=p.get_shape())
                       for p in self.params]
    self.param_assign_ops = [p.assign(new_p)
                             for p, new_p in zip(self.params, self.new_params)]
    self.opt_params = self.trainer.variables()
    self.new_opt_params = [tf.placeholder(p.dtype, shape=p.get_shape())
                           for p in self.opt_params]
    self.opt_param_assign_ops = [
      p.assign(new_p) for p, new_p in zip(self.opt_params, self.new_opt_params)
    ]
    self.reset_optimizer_op = tf.variables_initializer(
      self.trainer.variables() + self.burn_in_trainer.variables())

    self.loss_names = (list(self.loss_endpoints_names)
                       + ['clip_grad_norm', 'nonclip_grad_norm', 'param_norm'])

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
                       self.param_norm, self._train_batch],
        feed_dict=td_map
      )[0:len(self.loss_names)]

    def burn_in(lr, cliprange, lam, weights=None):
      td_map = _prepare_td_map(lr, cliprange, lam, weights)
      return self.sess.run(
        self.losses + [self.clip_grad_norm_vf, self.nonclip_grad_norm_vf,
                       self.param_norm, self._burn_in],
        feed_dict=td_map
      )[0:len(self.loss_names)]

    def save(save_path):
      ps = self.sess.run(self.params)
      joblib.dump(ps, save_path)

    def load_model(loaded_params):
      param_assign_ops = self.param_assign_ops[:len(loaded_params)]
      self.sess.run(
        param_assign_ops,
        feed_dict={p: v for p, v in zip(self.new_params, loaded_params)}
      )

    def restore_optimizer(loaded_opt_params):
      self.sess.run(self.opt_param_assign_ops,
                    feed_dict={p: v for p, v in zip(self.new_opt_params, loaded_opt_params)})

    def load(load_path):
      loaded_params = joblib.load(load_path)
      load_model(loaded_params)

    def read_params():
      return self.sess.run(self.params)

    def read_opt_params():
      return self.sess.run(self.opt_params)

    def reset():
      self.sess.run(self.reset_optimizer_op)

    self.train_batch = train_batch
    self.burn_in = burn_in
    self.save = save
    self.load_model = load_model
    self.restore_optimizer = restore_optimizer
    self.load = load
    self.read_params = read_params
    self.read_opt_params = read_opt_params
    self.reset = reset

  def _run_train_loop(self, nbatch):
    lr = as_func(self.task.hyperparam.learning_rate)
    cliprange = as_func(self.task.hyperparam.cliprange)
    lam = self.task.hyperparam.lam  # lambda for the td-lambda term
    weights = None
    if self.rwd_shape:
      assert hasattr(self.task.hyperparam, 'reward_weights')
      weights = np.array(self.task.hyperparam.reward_weights, dtype=np.float32)
      if len(weights.shape) == 1:
        weights = np.expand_dims(weights, 0)
    self.total_timesteps = getattr(self.task.hyperparam, 'total_timesteps',
                                   self.total_timesteps)
    burn_in_timesteps = 0
    if self._need_burn_in:
      burn_in_timesteps = getattr(self.task.hyperparam, 'burn_in_timesteps',
                                  self.burn_in_timesteps)
    nupdates_burn_in = int(burn_in_timesteps // nbatch)
    nupdates = nupdates_burn_in + int(self.total_timesteps // nbatch)
    mblossvals = []
    tfirststart = time.time()
    tstart = time.time()
    total_samples = self._data_server.unroll_num * self.unroll_length
    logger.log('Start Training')
    for update in xrange(1, nupdates + 1):
      frac = 1.0 - (update - 1.0) / nupdates
      lrnow = lr(frac)
      cliprangenow = cliprange(frac)
      if update <= nupdates_burn_in:
        mblossvals.append(self.burn_in(lrnow, cliprangenow, lam, weights))
      else:
        mblossvals.append(self.train_batch(lrnow, cliprangenow, lam, weights))
      # publish models
      if update % self.pub_interval == 0 and self.should_push_model:
        self._model_pool_apis.push_model(
          self.read_params(), self.task.hyperparam, self.model_key,
          learner_meta=self.read_opt_params()
        )
      # logging stuff
      if update % self.log_interval == 0 or update == 1:
        lossvals = np.mean(mblossvals, axis=0)
        mblossvals = []
        tnow = time.time()
        consuming_fps = int(
          nbatch * min(update, self.log_interval) / (tnow - tstart)
        )
        time_elapsed = tnow - tfirststart
        total_samples_now = self._data_server.unroll_num * self.unroll_length
        receiving_fps = (total_samples_now - total_samples) / (tnow - tstart)
        total_samples = total_samples_now
        tstart = time.time()
        # 'scope_name/var' style for grouping Tab in Tensorboard webpage
        # lp is short for Learning Period
        scope = 'lp{}/'.format(self._lrn_period_count)
        logger.logkvs({
          scope + "lrn_period_count": self._lrn_period_count,
          scope + "burn_in_value": update <= nupdates_burn_in,
          scope + "nupdates": update,
          scope + "total_timesteps": update * nbatch,
          scope + "all_consuming_fps": consuming_fps,
          scope + 'time_elapsed': time_elapsed,
          scope + "total_samples": total_samples,
          scope + "receiving_fps": receiving_fps,
          scope + "aband_samples": (self._data_server.aband_unroll_num *
                                    self.unroll_length)
          })
        logger.logkvs({scope + lossname: lossval for lossname, lossval
                       in zip(self.loss_names, lossvals)})
        logger.dumpkvs()
      if self.save_interval and (
          update % self.save_interval == 0 or update == 1) and logger.get_dir():
        checkdir = osp.join(logger.get_dir(), 'checkpoints')
        os.makedirs(checkdir, exist_ok=True)
        savepath = osp.join(checkdir, '%.5i' % update)
        logger.log('Saving log to', savepath)
        self.save(savepath)
    if self.should_push_model:
      self._model_pool_apis.push_model(
        self.read_params(), self.task.hyperparam, self.model_key,
        learner_meta=self.read_opt_params())
    return

  def _init_task(self):
    task = self.task
    logger.log('Period: {},'.format(self._lrn_period_count),
               'Task: {}'.format(str(task)))
    logger.log('Continue training from model: {}. New model id: {}.'.format(
      str(task.parent_model_key), str(self.model_key)))

    hyperparam = task.hyperparam
    if task.parent_model_key is None:
      logger.log(
        'Parent model is None, '
        'pushing new model {} params to ModelPool'.format(self.model_key)
      )
      self._model_pool_apis.push_model(self.read_params(), hyperparam,
                                       self.model_key,
                                       learner_meta=self.read_opt_params())
    elif self.model_key != self.last_model_key:
      logger.log(
        'Parent model {} exists, pulling its params from ModelPool '
        'as new model {}'.format(task.parent_model_key, self.model_key)
      )
      model_obj = self._model_pool_apis.pull_model(task.parent_model_key)
      self._model_pool_apis.push_model(model_obj.model, hyperparam,
                                       self.model_key,
                                       learner_meta=self.read_opt_params())
      self.load_model(model_obj.model)
      learner_meta = self._model_pool_apis.pull_learner_meta(
        task.parent_model_key)
      if learner_meta is not None:
        logger.log(
          'Restore optimizer from model {}'.format(task.parent_model_key)
        )
        self.restore_optimizer(learner_meta)
      else:
        self.reset()
    else:
      logger.log('Continue training model {}.'.format(self.model_key))

  @staticmethod
  def clip_grads_vars(grads_and_vars, all_clip_vars, max_grad_norm):
    nonclip_grads_and_vars = [gv for gv in grads_and_vars if
                              gv[1] not in all_clip_vars]
    nonclip_grad_norm = tf.global_norm(
      [grad for grad, _ in nonclip_grads_and_vars])
    clip_grads_and_vars = [gv for gv in grads_and_vars if
                           gv[1] in all_clip_vars]
    if len(clip_grads_and_vars) > 0:
      clip_grads, clip_vars = zip(
        *[gv for gv in grads_and_vars if gv[1] in all_clip_vars])
    else:
      clip_grads, clip_vars = [], []
    if max_grad_norm is not None:
      clip_grads, clip_grad_norm = tf.clip_by_global_norm(clip_grads,
                                                          max_grad_norm)
      clip_grads_and_vars = list(zip(clip_grads, clip_vars))
    else:
      clip_grad_norm = tf.global_norm(clip_grads)
      clip_grads_and_vars = list(zip(clip_grads, clip_vars))
    grads_and_vars = clip_grads_and_vars + nonclip_grads_and_vars
    return grads_and_vars, clip_grad_norm, nonclip_grad_norm

  def build_loss(self, model, input_data):
    entropy_list = nest.flatten(model.loss.entropy_loss)
    if isinstance(self.ent_coef, list):
      assert len(entropy_list) == len(
        self.ent_coef), 'Lengths of ent and ent_coef mismatch.'
      print('ent_coef: {}'.format(self.ent_coef))
      entropy = tf.reduce_sum(
        [e * ec for e, ec in zip(entropy_list, self.ent_coef)])
    else:
      entropy = tf.reduce_sum(entropy_list) * self.ent_coef
    distill_loss = tf.constant(0, dtype=tf.float32)
    if self.distillation:
      distill_losses = nest.flatten(model.loss.distill_loss)
      if isinstance(self.distill_coef, list):
        assert len(distill_losses) == len(
          self.distill_coef), 'Lengths of distill and distill_coef mismatch.'
        print('distill_coef: {}'.format(self.distill_coef))
        distill_loss = tf.reduce_sum(
          [d * dc for d, dc in zip(distill_losses, self.distill_coef)])
      else:
        distill_loss = tf.reduce_sum(distill_losses) * self.distill_coef
    if isinstance(self.vf_coef, list):
      value_shape = model.loss.value_loss.shape
      assert len(value_shape) == 1 and value_shape[0] == len(self.vf_coef)
      print('vf_coef: {}'.format(self.vf_coef))
      value_loss = tf.reduce_sum(
        model.loss.value_loss * tf.constant(self.vf_coef))
    else:
      value_loss = tf.reduce_sum(model.loss.value_loss) * self.vf_coef
    ep_loss = tf.constant(0, dtype=tf.float32)
    for loss_name, loss_coef in self.ep_loss_coef.items():
      ep_loss += model.loss.loss_endpoints[loss_name] * loss_coef
    loss = (model.loss.pg_loss + value_loss - entropy + distill_loss + ep_loss)
    return loss, value_loss, model.loss.loss_endpoints.values()