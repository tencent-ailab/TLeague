from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import time
import warnings

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

from tleague.learners.base_learner import BaseLearner
from tleague.learners.data_server_gail import DataServerGAIL
from tleague.learners.data_server_v3 import DataServer
from tleague.utils import logger
from tleague.utils.data_structure import DiscExpertData
from tleague.utils.data_structure import DiscAgentData


def as_func(obj):
  if isinstance(obj, float):
    return lambda x: obj
  else:
    assert callable(obj)
    return obj


class GAILLearner(BaseLearner):
  def __init__(self, league_mgr_addr, model_pool_addrs, learner_ports,
               rm_size, batch_size, gpu_id,
               agent_ob_space, agent_ac_space, expert_ob_space, expert_ac_space,
               discriminator, discriminator_config, disc_grad_penalty_coef=0.0,
               max_grad_norm=0.5, pub_interval=500, log_interval=100,
               save_interval=0, total_timesteps=5e7, learner_id='',
               batch_worker_num=4, prefetch_buffer_size=2,
               unroll_length=32, rollout_length=1, log_dir="",
               use_mixed_precision=False, use_sparse_as_dense=True,
               adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-5,
               log_infos_interval=20, ep_loss_coef=None,
               data_path=None, expert_type='batch',
               **kwargs):

    if len(kwargs) > 0:
      for k in kwargs:
        warnings.warn('Unused args passed in Learner: {}'.format(k))
    super(GAILLearner, self).__init__(league_mgr_addr, model_pool_addrs,
                                      learner_ports, learner_id)
    self.LR = tf.placeholder(tf.float32, [])
    """Learning Rate"""

    self.ep_loss_coef = ep_loss_coef or {}
    """Coefficients for those losses from the endpoints."""

    self._init_const(total_timesteps, batch_size, unroll_length,
                     pub_interval, log_interval, log_dir, save_interval,
                     discriminator, disc_grad_penalty_coef,
                     discriminator_config, rollout_length, max_grad_norm)

    # allow_soft_placement=True can fix issue when some op cannot be defined on
    # GPUs for tf-1.8.0; tf-1.13.1 does not have this issue
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(gpu_id)
    self.sess = tf.Session(config=config)
    self.use_hvd = has_hvd and hvd.size() > 1
    self.rank = hvd.rank() if self.use_hvd else 0

    # prepare net config
    net_config = discriminator.net_config_cls(agent_ob_space,
                                              agent_ac_space,
                                              **discriminator_config)

    # Prepare dataset
    ds_agent = DiscAgentData(agent_ob_space, agent_ac_space)
    if expert_type == 'batch':
      # if use batch expert, the expert data are stored offline and the
      # expert data server directly reads data from the files
      ds_expert = DiscExpertData(net_config.n_feature)
      self._data_server_agent = DataServer(
        learner_ports, rm_size,
        unroll_length, batch_size, ds_agent,
        gpu_id_list=(0,),
        batch_worker_num=batch_worker_num,
        rollout_length=rollout_length,
        prefetch_buffer_size=prefetch_buffer_size,
        log_infos_interval=log_infos_interval)
      self._data_server_expert = DataServerGAIL(
        batch_size, ds_expert,
        gpu_id_list=(0,),
        batch_worker_num=batch_worker_num,
        prefetch_buffer_size=prefetch_buffer_size,
        data_path=data_path)
    elif expert_type == 'online':
      # for online expert, the data server needs to bind online expert actors
      # the first port is for publish; others are for receiving
      ds_expert = DiscAgentData(expert_ob_space, expert_ac_space)
      agent_ports = learner_ports[0:1] + learner_ports[
                                         1:(len(learner_ports) - 1) // 2 + 1]
      expert_ports = learner_ports[0:1] + learner_ports[
                                          -(len(learner_ports) - 1) // 2:]
      assert len(agent_ports) == len(expert_ports)
      self._data_server_agent = DataServer(
        agent_ports, rm_size,
        unroll_length, batch_size, ds_agent,
        gpu_id_list=(0,),
        batch_worker_num=batch_worker_num,
        rollout_length=rollout_length,
        prefetch_buffer_size=prefetch_buffer_size,
        log_infos_interval=log_infos_interval)
      self._data_server_expert = DataServer(
        expert_ports, rm_size,
        unroll_length, batch_size, ds_expert,
        gpu_id_list=(0,),
        batch_worker_num=batch_worker_num,
        rollout_length=rollout_length,
        prefetch_buffer_size=prefetch_buffer_size,
        log_infos_interval=log_infos_interval)
    else:
      raise NotImplementedError('Unknown data_server_type for GAIL.')

    # build the model net
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as model_scope:
      pass

    def create_model(inputs, nc):
      return discriminator.net_build_fun(inputs=inputs, nc=nc,
                                         scope=model_scope)

    device = '/gpu:{}'.format(0)
    with tf.device(device):
      input_data = (self._data_server_agent.input_datas[0],
                    self._data_server_expert.input_datas[0])
      if 'use_xla' in discriminator_config and discriminator_config['use_xla']:
        try:
          # Use tensorflow's accerlated linear algebra compile method
          with tf.xla.experimental.jit_scope(True):
            self.model = create_model(input_data, net_config)
        except:
          logger.log("WARNING: using tf.xla requires tf version>=1.15.")
          self.model = create_model(input_data, net_config)
      else:
        self.model = create_model(input_data, net_config)
      self.build_loss(self.model)
    if self.use_hvd:
      self.losses = [hvd.allreduce(loss) for loss in self.losses]
    else:
      self.losses = list(self.losses)
    self.params = tf.trainable_variables(scope='model')
    self.param_norm = tf.global_norm(self.params)

    self.trainer = tf.train.AdamOptimizer(learning_rate=self.LR,
                                          beta1=adam_beta1,
                                          beta2=adam_beta2,
                                          epsilon=adam_eps)
    if use_mixed_precision:
      try:
        self.trainer = \
          tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(
            self.trainer)
      except:
        logger.warn("using tf mixed_precision requires tf version>=1.15.")
    if self.use_hvd:
      self.trainer = hvd.DistributedOptimizer(
        self.trainer, sparse_as_dense=use_sparse_as_dense)

    if 'use_lstm' in discriminator_config and discriminator_config['use_lstm']:
      self.clip_vars = self.model.vars.lstm_vars
    else:
      self.clip_vars = []
    self._build_train_op()

    self.loss_endpoints_names = self.model.loss.loss_endpoints.keys()
    if self.use_hvd:
      barrier_op = hvd.allreduce(tf.Variable(0.))
      broadcast_op = hvd.broadcast_global_variables(0)
    tf.global_variables_initializer().run(session=self.sess)
    self._build_ops()
    self.sess.graph.finalize()

    self.barrier = lambda: self.sess.run(barrier_op) if self.use_hvd else None
    self.broadcast = lambda: self.sess.run(
      broadcast_op) if self.use_hvd else None
    self.broadcast()
    # logging stuff
    format_strs = (['stdout', 'log', 'tensorboard', 'csv'] if self.rank == 0
                   else ['stdout', 'log', 'tensorboard', 'csv'])
    dir = os.path.join(self.log_dir, f'{self._learner_id}rank{self.rank}')
    logger.configure(dir=dir, format_strs=format_strs)

  def _build_train_op(self):
    grads_and_vars = self.trainer.compute_gradients(self.loss, self.params)

    grads_and_vars, self.clip_grad_norm, self.nonclip_grad_norm = \
      self.clip_grads_vars(grads_and_vars, self.clip_vars, self.max_grad_norm)

    self._train_batch = self.trainer.apply_gradients(grads_and_vars)

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
    self._data_server_agent._update_model_id(self.model_key)
    if hasattr(self, '_data_server_expert'):
      self._data_server_expert._update_model_id(self.model_key)
    # Use different model, clear the replay memory
    if (self.last_model_key is None
        or self.last_model_key != self.task.parent_model_key):
      self._data_server_agent.reset()
      if hasattr(self, '_data_server_expert'):
        self._data_server_expert.reset()
      while not (self._data_server_agent.ready_for_train & (
          self._data_server_expert.ready_for_train if
          hasattr(self, '_data_server_expert') else True)):
        time.sleep(5)
        self._model_pool_apis.push_model(
          self.read_params(), self.task.hyperparam, self.model_key,
          learner_meta=self.read_opt_params()
        )

    self.barrier()
    nbatch = self.batch_size * hvd.size() if self.use_hvd else self.batch_size
    self.should_push_model = (self.rank == 0)
    self._run_train_loop(nbatch)

  def _init_const(self, total_timesteps, batch_size, unroll_length,
                  pub_interval, log_interval, log_dir, save_interval,
                  model, disc_grad_penalty_coef,
                  model_config, rollout_length, max_grad_norm):
    self.total_timesteps = total_timesteps
    self._train_batch = []
    self.batch_size = batch_size
    self.unroll_length = unroll_length
    self.pub_interval = pub_interval
    self.log_interval = log_interval
    self.log_dir = log_dir
    self.save_interval = save_interval
    self.model = model
    self.disc_grad_penalty_coef = disc_grad_penalty_coef,
    self.rnn = (False if 'use_lstm' not in model_config
                else model_config['use_lstm'])
    self.hs_len = None
    model_config['batch_size'] = batch_size
    model_config['rollout_len'] = rollout_length
    if self.rnn:
      self.hs_len = model_config['hs_len']
    self.max_grad_norm = max_grad_norm

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
      self.trainer.variables())

    self.loss_names = (list(self.loss_endpoints_names)
                       + ['clip_grad_norm', 'nonclip_grad_norm', 'param_norm'])

    def _prepare_map(lr):
      feed_map = {self.LR: lr}
      return feed_map

    def train_batch(lr):
      td_map = _prepare_map(lr)
      return self.sess.run(
        self.losses + [self.clip_grad_norm, self.nonclip_grad_norm,
                       self.param_norm, self._train_batch],
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
                    feed_dict={p: v for p, v in
                               zip(self.new_opt_params, loaded_opt_params)})

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
    self.save = save
    self.load_model = load_model
    self.restore_optimizer = restore_optimizer
    self.load = load
    self.read_params = read_params
    self.read_opt_params = read_opt_params
    self.reset = reset

  def _run_train_loop(self, nbatch):
    lr = as_func(self.task.hyperparam.learning_rate)
    self.total_timesteps = getattr(self.task.hyperparam, 'total_timesteps',
                                   self.total_timesteps)

    nupdates = int(self.total_timesteps // nbatch)
    mblossvals = []
    tfirststart = time.time()
    tstart = time.time()
    total_agent_samples = self._data_server_agent.unroll_num * \
                          self.unroll_length
    total_expert_samples = self._data_server_expert.unroll_num * \
                           self.unroll_length
    logger.log('Start Training')
    for update in xrange(1, nupdates + 1):
      frac = 1.0 - (update - 1.0) / nupdates
      lrnow = lr(frac)
      mblossvals.append(self.train_batch(lrnow))
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
        total_agent_samples_now = self._data_server_agent.unroll_num * \
                                  self.unroll_length
        total_expert_samples_now = self._data_server_expert.unroll_num * \
                                   self.unroll_length
        agent_receiving_fps = (total_agent_samples_now -
                               total_agent_samples) / (tnow - tstart)
        expert_receiving_fps = (total_expert_samples_now -
                                total_expert_samples) / (tnow - tstart)
        total_agent_samples = total_agent_samples_now
        total_expert_samples = total_expert_samples_now
        tstart = time.time()
        # 'scope_name/var' style for grouping Tab in Tensorboard webpage
        # lp is short for Learning Period
        scope = 'lp{}/'.format(self._lrn_period_count)
        logger.logkvs({
          scope + "lrn_period_count": self._lrn_period_count,
          scope + "nupdates": update,
          scope + "total_timesteps": update * nbatch,
          scope + "all_consuming_fps": consuming_fps,
          scope + 'time_elapsed': time_elapsed,
          scope + "total_samples": total_agent_samples,
          scope + "receiving_fps(agent,-)": agent_receiving_fps,
          scope + "receiving_fps(expert,+)": expert_receiving_fps,
          scope + "aband_samples(-)": (
                self._data_server_agent.aband_unroll_num *
                self.unroll_length),
          scope + "aband_samples(+)": (
                self._data_server_expert.aband_unroll_num *
                self.unroll_length),
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

  def build_loss(self, model):
    disc_grad_penalty_loss = tf.reduce_sum(
      model.loss.disc_grad_penalty_loss) * self.disc_grad_penalty_coef
    ep_loss = tf.constant(0, dtype=tf.float32)
    for loss_name, loss_coef in self.ep_loss_coef.items():
      ep_loss += model.loss.loss_endpoints[loss_name] * loss_coef
    self.loss = (model.loss.disc_loss + disc_grad_penalty_loss + ep_loss)
    self.losses = model.loss.loss_endpoints.values()
