from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
from functools import partial
from multiprocessing import Pool

from absl import app
from absl import flags

from . import get_worker_addr
from . import get_worker_ports
from . import group_workers_by_ip
from . import pair_actors_learners
from . import parse_workers_spec_csv
from . import split_workers
from . import to_cmd_str
from . import zip_containing_folder_relpath
from tleague.scripts import _func_run_cmd
from tleague.scripts import make_quote_cmd


FLAGS = flags.FLAGS
flags.DEFINE_string('clust_spec_csv_path', '../sandbox/clust_spec_example.csv',
                    'cluster specification csv file path.')
flags.DEFINE_string('remote_working_folder', '/root/code',
                    'unified remote working folder across all machines.')
flags.DEFINE_boolean('force_overwrite_remote', True,
                     'Should force overwriting remote working folder?')
flags.DEFINE_string('tmux_sess', 'tlea', 'tmux session name for each machine')
flags.DEFINE_string('local_worker_pre_cmd', '',
                    "pre command for each LOCAL worker. "
                    "e.g., activate python virutal env")
flags.DEFINE_string('remote_worker_pre_cmd', '',
                    "pre command for each REMOTE worker. "
                    "e.g., activate python virutal env")
flags.DEFINE_string('python_bin', 'python',
                    "python binary name or full path"
                    "e.g., python, /usr/bin/python, python3")
flags.DEFINE_integer('n_process', 1,
                     'number of parallel processes for connections.')
# actor-learner common
flags.DEFINE_integer("unroll_length", 32, "unroll length")
flags.DEFINE_integer("n_v", 1, "value length")
flags.DEFINE_string("env", "sc2", "task environment name")
# actor
flags.DEFINE_integer("actor_update_model_freq", 32,
                     "update model every n steps")
flags.DEFINE_integer("actor_verbose", 21,
                     "verbosity. The smaller, the noisier. Reference:"
                     "10:DEBUG, 20:INFO, 30: WARN, 40: ERROR, 50:DISABLED")
flags.DEFINE_integer("actor_log_interval_steps", 51,
                     "frequency of printing log in steps")
flags.DEFINE_boolean("actor_rwd_shape", True, "do reward shape in actor")
flags.DEFINE_boolean("actor_distillation", False, "send logits to learner")
# learner
flags.DEFINE_integer("lrn_split_num", 1, "Depreacted")
flags.DEFINE_integer("lrn_rollout_length", 1, "sample n frames consecutively.")
flags.DEFINE_integer("batch_size", 4096, "Num of samples in each batch")
flags.DEFINE_integer("lrn_rm_size", 64000, "Num of samples in replay memory")
flags.DEFINE_integer("lrn_pub_interval", 500,
                     "freq of pub model to actors, in num of batch")
flags.DEFINE_integer("lrn_log_interval", 100,
                     "freq of print log, in num of batch")
flags.DEFINE_integer("lrn_total_timesteps", 50000000,
                     "total time steps per learning period, "
                     "i.e., steps before adding the current model to the pool.")
flags.DEFINE_integer("lrn_burn_in_timesteps", 0,
                     "total time steps for each learning period to burn in "
                     "at beginning.")
flags.DEFINE_integer("lrn_pull_worker_num", 2,
                     "pull data worker number for each pull data socket")
flags.DEFINE_integer("lrn_batch_worker_num", 4,
                     "sample data worker number for each dataset")
flags.DEFINE_boolean('lrn_adv_normalize', True,
                     'Whether normalize adv for ppo')
flags.DEFINE_boolean("lrn_rwd_shape", False, "do reward shape in learner")
flags.DEFINE_boolean("lrn_merge_pi", False, "add up neglogp in learner")
flags.DEFINE_string("learner", "PPOLearner", "learner used")
flags.DEFINE_string("policy", "DeepMultiHeadMlpPolicy", "policy used")
flags.DEFINE_string("policy_config",
                    "{}", "config used for policy")
flags.DEFINE_string("learner_config",
                    "tleague.learners.ppo_configs.PPO_Config_v0",
                    "config used for learner")
flags.DEFINE_string("piecewise_fusion_schedule",
                    "-1",
                    "piecewise_fusion_schedule for policy_network")
flags.DEFINE_string("piecewise_fusion_schedule_vf",
                    "-1",
                    "piecewise_fusion_schedule for value_network")


# league_mgr
flags.DEFINE_string('mutable_hyperparam_type',
                    'MutableHyperparamRandPredefSC2V1',
                    "Mutable hyperparam class name. checkout "
                    "tleague.hyperparam_mgr.hyperparam_types for supported"
                    " classes.")
flags.DEFINE_string('hyperparam_config_name', "",
                    "Config used for Mutable hyperparam. "
                    "checkout tleague.hyperparam_mgr.configs.")
flags.DEFINE_string("game_mgr_type", "RefCountGameMgr",
                    "game_mgr class name. "
                    "checkout tleague.game_mgr.game_mgrs for supported "
                    "classes.")
flags.DEFINE_string("leagmgr_restore_checkpoint_dir", "",
                    "checkpoint dir from which the tleague restores. ")
flags.DEFINE_string("leagmgr_save_checkpoint_root", "",
                    "checkpoint root dir to which the tleague saves. "
                    "Will create sub folders during run.")
flags.DEFINE_integer("leagmgr_save_interval_secs", 4800,
                     "checkpoint saving frequency in seconds.")
flags.DEFINE_integer("leagmgr_verbose", 11,
                     "verbosity. The smaller, the noisier. Reference:"
                     "10:DEBUG, 20:INFO, 30: WARN, 40: ERROR, 50:DISABLED")
flags.DEFINE_boolean('mute_actor_msg', False,
                     'Whether print req/rep of actors')
flags.DEFINE_integer("pseudo_learner_num", 0,
                     "pseudo parallel run pseudo_learner_num learners "
                     "with fewer real learners")
flags.DEFINE_string("init_model_path", "", "initial model path")
# model pool
flags.DEFINE_integer("modelpool_verbose", 50,
                     "verbosity. The smaller, the noisier. Reference:"
                     "10:DEBUG, 20:INFO, 30: WARN, 40: ERROR, 50:DISABLED")
flags.DEFINE_boolean('reboot_actor_on_failure', False,
                     'Whether reboot when actor crushes.')


def _learner_cmd(league_mgr_addr, model_pool_addrs, learner_ports,
                 cuda_visible_devices):
  # one learner can uses several GPUs
  # cuda_visible_devices is seperated by ":", we need to convert it to ","
  # gpu_num 0 means *no GPU*
  # see run_pg_learner.py arg specs
  if cuda_visible_devices:
    cuda_visible_devices = cuda_visible_devices.split(':')
    gpu_num = len(cuda_visible_devices)
    cuda_visible_devices = ','.join(cuda_visible_devices)
  else:
    gpu_num = 0
    cuda_visible_devices = "-1"
  return [
    'CUDA_VISIBLE_DEVICES={}'.format(cuda_visible_devices),
    FLAGS.python_bin, '-m',
    'tleague.scripts.run_pg_learner',
    '--league_mgr_addr', league_mgr_addr,
    '--model_pool_addrs', model_pool_addrs,
    '--learner_ports', learner_ports,
    '--gpu_num', gpu_num,
    '--unroll_length', FLAGS.unroll_length,
    '--batch_size', FLAGS.batch_size,
    '--rm_size', FLAGS.lrn_rm_size,
    '--pub_interval', FLAGS.lrn_pub_interval,
    '--log_interval', FLAGS.lrn_log_interval,
    '--total_timesteps', FLAGS.lrn_total_timesteps,
    '--burn_in_timesteps', FLAGS.lrn_burn_in_timesteps,
    '--env', FLAGS.env,
    '--policy', FLAGS.policy,
    '--policy_config', make_quote_cmd(FLAGS.policy_config),
    '--adv_normalize' if FLAGS.lrn_adv_normalize else '--noadv_normalize',
    '--learner', FLAGS.learner,
    '--n_v', FLAGS.n_v,
    '--rollout_length', FLAGS.lrn_rollout_length,
    '--rwd_shape' if FLAGS.lrn_rwd_shape else '--norwd_shape',
    '--merge_pi' if FLAGS.lrn_merge_pi else '--nomerge_pi',
    '--pull_worker_num', FLAGS.lrn_pull_worker_num,
    '--batch_worker_num', FLAGS.lrn_batch_worker_num,
    '--learner_config', make_quote_cmd(FLAGS.learner_config),
    '--type', 'PPO',
  ]


def _actor_cmd(league_mgr_addr, model_pool_addrs, learner_addr):
  # should never use GPU
  return [
    'CUDA_VISIBLE_DEVICES=',
    FLAGS.python_bin, '-m',
    'tleague.scripts.run_pg_actor',
    '--league_mgr_addr', league_mgr_addr,
    '--model_pool_addrs', model_pool_addrs,
    '--learner_addr', learner_addr,
    '--unroll_length', FLAGS.unroll_length,
    '--update_model_freq', FLAGS.actor_update_model_freq,
    '--env', FLAGS.env,
    '--policy', FLAGS.policy,
    '--policy_config', make_quote_cmd(FLAGS.policy_config),
    '--verbose', FLAGS.actor_verbose,
    '--log_interval_steps', FLAGS.actor_log_interval_steps,
    '--n_v', FLAGS.n_v,
    '--rwd_shape' if FLAGS.actor_rwd_shape else '--norwd_shape',
    '--reboot_on_failure' if FLAGS.reboot_actor_on_failure \
        else '--noreboot_on_failure',
    '--distillation' if FLAGS.actor_distillation else '--nodistillation',
    '--type', 'PPO'
  ]


def _league_mgr_cmd(model_pool_addrs, port):
  # should never use GPU
  return [
    'CUDA_VISIBLE_DEVICES=',
    FLAGS.python_bin, '-m',
    'tleague.scripts.run_league_mgr',
    '--model_pool_addrs', model_pool_addrs,
    '--port', port,
    '--mutable_hyperparam_type', FLAGS.mutable_hyperparam_type,
    '--game_mgr_type', FLAGS.game_mgr_type,
    '--hyperparam_config_name=' + make_quote_cmd(FLAGS.hyperparam_config_name),
    '--restore_checkpoint_dir=' + FLAGS.leagmgr_restore_checkpoint_dir,
    '--save_checkpoint_root=' + FLAGS.leagmgr_save_checkpoint_root,
    '--save_interval_secs', FLAGS.leagmgr_save_interval_secs,
    '--mute_actor_msg' if FLAGS.mute_actor_msg else '--nomute_actor_msg',
    '--verbose', FLAGS.leagmgr_verbose,
    '--pseudo_learner_num', FLAGS.pseudo_learner_num,
    '--init_model_path=' + FLAGS.init_model_path,
  ]


def _model_pool_cmd(ports):
  # should never use GPU
  return [
    'CUDA_VISIBLE_DEVICES=',
    FLAGS.python_bin, '-m',
    'tleague.scripts.run_model_pool',
    '--ports', ports,
    '--verbose', FLAGS.modelpool_verbose,
  ]


def create_worker_cmd(job_to_workers, act_addr_to_lrn, worker):
  league_mgr_addr = get_worker_addr(job_to_workers['league_mgr'][0])
  model_pool_addrs = ','.join(get_worker_addr(w) for w
                              in job_to_workers['model_pool'])

  if worker.job == 'learner':
    cmds = _learner_cmd(league_mgr_addr, model_pool_addrs,
                        learner_ports=get_worker_ports(worker),
                        cuda_visible_devices=worker.cuda_visible_devices)
  elif worker.job == 'actor':
    learner_addr = get_worker_addr(act_addr_to_lrn[get_worker_addr(worker)])
    cmds = _actor_cmd(league_mgr_addr, model_pool_addrs, learner_addr)
  elif worker.job == 'league_mgr':
    cmds = _league_mgr_cmd(model_pool_addrs, port=get_worker_ports(worker))
  elif worker.job == 'model_pool':
    cmds = _model_pool_cmd(ports=get_worker_ports(worker))
  else:
    raise ValueError('unknown job %s'.format(worker.job))
  return to_cmd_str(cmds)


def main(_):
  workers = parse_workers_spec_csv(FLAGS.clust_spec_csv_path)
  job_to_workers = split_workers(workers)
  ip_to_workers = group_workers_by_ip(workers)
  act_addr_to_lrn = pair_actors_learners(actors=job_to_workers['actor'],
                                         learners=job_to_workers['learner'])
  zip_file_path = zip_containing_folder_relpath()

  q_failed_workers = multiprocessing.Manager().Queue()
  mp = Pool(processes=FLAGS.n_process)
  _worker_cmd = partial(create_worker_cmd, job_to_workers, act_addr_to_lrn)
  tmp_func = partial(_func_run_cmd, _worker_cmd,
                     zip_file_path, q_failed_workers,
                     FLAGS.remote_working_folder, FLAGS.force_overwrite_remote,
                     FLAGS.remote_worker_pre_cmd, FLAGS.local_worker_pre_cmd,
                     FLAGS.python_bin, FLAGS.tmux_sess)
  mp.map(tmp_func, list(ip_to_workers.values()))
  mp.close()
  mp.join()

  n_sucess = len(workers) - q_failed_workers.qsize()
  n_failure = q_failed_workers.qsize()
  for _ in range(q_failed_workers.qsize()):
    w, e = q_failed_workers.get()
    print(w)
    print('{}: {}'.format(type(e), e))
  print('\nall commands sent. Succeeded: {}, Failed: {}'.format(
    n_sucess, n_failure
  ))


if __name__ == '__main__':
  app.run(main)
