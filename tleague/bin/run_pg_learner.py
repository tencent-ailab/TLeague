from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
  import horovod.tensorflow as hvd
  has_hvd = True
except:
  has_hvd = False
from absl import app
from absl import flags
from tleague.envs.create_envs import env_space
from tleague.learners.ppo_learner3 import PPOLearner
from tleague.learners.ppo2_learner import PPO2Learner
from tleague.learners.vtrace_learner import VtraceLearner
from tleague.utils import read_config_dict
from tleague.utils import import_module_or_data


FLAGS = flags.FLAGS
flags.DEFINE_string("league_mgr_addr", "localhost:10005",
                    "League manager address.")
flags.DEFINE_string("model_pool_addrs", "localhost:10003:10004",
                    "Model Pool address.")
flags.DEFINE_multi_string(
  "learner_spec", [
    "0:10001:10002,1:10003:10004",
    "0:10001:10002,2:10003:10004,3:10005:10006",
    ],
  "Comma separated gpu_id:port1:port2 triplet, which can occur as many times as"
  " the number of GPUs used in a host. This arg can occur as many times as the"
  " number of remote machines (in different IPs).",
  short_name='l'
)
flags.DEFINE_string("learner_id", "",
                    "Learner id")

# RL related
flags.DEFINE_integer("unroll_length", 32, "trajectory unroll length")
flags.DEFINE_integer("rollout_length", 1,
                     "sample n frames consecutively for RNN/LSTM.")
flags.DEFINE_integer("batch_size", 4096, "Num of samples in each batch")
flags.DEFINE_integer("rm_size", 64000, "Num of samples in replay memory")
flags.DEFINE_integer("pub_interval", 500,
                     "freq of pub model to actors, in num of batch")
flags.DEFINE_integer("log_interval", 100, "freq of print log, in num of batch")
flags.DEFINE_integer("save_interval", 0, "freq of save model, in num of batch")
flags.DEFINE_integer("batch_worker_num", 4, "batch_worker_num")
flags.DEFINE_integer("pull_worker_num", 2, "pull_worker_num")
flags.DEFINE_integer("total_timesteps", 50000000,
                     "total time steps per learning period, "
                     "i.e., steps before adding the current model to the pool.")
flags.DEFINE_integer("burn_in_timesteps", 0,
                     "total time steps for each learning period to burn in at beginning.")
flags.DEFINE_string("env", "sc2", "task env")
flags.DEFINE_string("env_config", "",
                    "python dict config used for env. "
                    "e.g., {'replay_dir': '/root/replays/ext471_zvz'}")
flags.DEFINE_string("interface_config", "",
                    "python dict config used for Arena interface. "
                    "e.g., {'zstat_data_src_path': '/root/replay_ds/zstat'}")
flags.DEFINE_string("policy", "tpolicies.ppo.policies.DeepMultiHeadMlpPolicy",
                    "policy used")
flags.DEFINE_string("policy_config", "{}", "config used for policy")
flags.DEFINE_string("post_process_data", None,
                    "post process of (X, A), drop useless mask in SC2.")
flags.DEFINE_boolean("rwd_shape", False, "do reward shape in learner")
flags.DEFINE_string("learner_config",
                    "tleague.learners.ppo_configs.PPO_Config_v0",
                    "config used for learner. py module or py expression.")
flags.DEFINE_string("type", "PPO", "PPO|PPO2|Vtrace learner type")
flags.DEFINE_string("data_server_version", "v1", "v2|v1")
flags.DEFINE_boolean("decode", False, "Store decoded/encoded samples in replay memory.")
flags.DEFINE_integer("log_infos_interval", 20,
                     "print infos every interval games")


def main(_):
  if has_hvd:
    hvd.init()

  all_learner_specs = []
  for item in FLAGS.learner_spec:
    all_learner_specs += item.split(',')
  this_learner_ind = 0 if not has_hvd else hvd.rank()
  local_learner_spec = all_learner_specs[this_learner_ind]
  gpu_id_ports = local_learner_spec.split(':')  # gpu_id:port1:port2
  gpu_id, learner_ports = int(gpu_id_ports[0]), gpu_id_ports[1:]

  env_config = read_config_dict(FLAGS.env_config)
  interface_config = read_config_dict(FLAGS.interface_config)
  ob_space, ac_space = env_space(FLAGS.env, env_config, interface_config)
  if FLAGS.post_process_data is not None:
    post_process_data = import_module_or_data(FLAGS.post_process_data)
    ob_space, ac_space = post_process_data(ob_space, ac_space)
  policy = import_module_or_data(FLAGS.policy)
  policy_config = read_config_dict(FLAGS.policy_config)
  learner_config = read_config_dict(FLAGS.learner_config)
  if FLAGS.type == 'PPO':
    Learner = PPOLearner
  elif FLAGS.type == 'PPO2':
    Learner = PPO2Learner
  else:
    Learner = VtraceLearner
  learner = Learner(league_mgr_addr=FLAGS.league_mgr_addr,
                    model_pool_addrs=FLAGS.model_pool_addrs.split(','),
                    gpu_id=gpu_id,
                    learner_ports=learner_ports,
                    unroll_length=FLAGS.unroll_length,
                    rm_size=FLAGS.rm_size,
                    batch_size=FLAGS.batch_size,
                    ob_space=ob_space, ac_space=ac_space,
                    pub_interval=FLAGS.pub_interval,
                    log_interval=FLAGS.log_interval,
                    save_interval=FLAGS.save_interval,
                    total_timesteps=FLAGS.total_timesteps,
                    burn_in_timesteps=FLAGS.burn_in_timesteps,
                    policy=policy,
                    policy_config=policy_config,
                    rwd_shape=FLAGS.rwd_shape,
                    learner_id=FLAGS.learner_id,
                    batch_worker_num=FLAGS.batch_worker_num,
                    pull_worker_num=FLAGS.pull_worker_num,
                    rollout_length=FLAGS.rollout_length,
                    data_server_version=FLAGS.data_server_version,
                    decode=FLAGS.decode,
                    log_infos_interval=FLAGS.log_infos_interval,
                    **learner_config)
  learner.run()


if __name__ == '__main__':
  app.run(main)
