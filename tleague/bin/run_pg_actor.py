from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

from absl import app
from absl import flags

from tleague.actors.ppo_actor import PPOActor
from tleague.actors.ppo2_actor import PPO2Actor
from tleague.actors.vtrace_actor import VtraceActor
from tleague.actors.ddpg_actor import DDPGActor
from tleague.envs.create_envs import create_env
from tleague.utils import read_config_dict
from tleague.utils import import_module_or_data
from tleague.utils import kill_sc2_processes_v2


FLAGS = flags.FLAGS
flags.DEFINE_string("league_mgr_addr", "localhost:10005",
                    "League manager address.")
flags.DEFINE_string("model_pool_addrs", "localhost:10003:10004",
                    "Model Pool address.")
flags.DEFINE_string("learner_addr", "localhost:10001:10002",
                    "Learner address")
# RL related
flags.DEFINE_integer("unroll_length", 32, "unroll length")
flags.DEFINE_integer("n_v", 1, "value length")
flags.DEFINE_integer("update_model_freq", 32, "update model every n steps")
flags.DEFINE_string("env", "sc2", "task env")
flags.DEFINE_string("env_config", "",
                    "python dict config used for env. "
                    "e.g., {'replay_dir': '/root/replays/ext471_zvz'}")
flags.DEFINE_string("interface_config", "",
                    "python dict config used for Arena interface. "
                    "e.g., {'zstat_data_src_path': '/root/replay_ds/zstat'}")
flags.DEFINE_string("policy", "tpolicies.ppo.policies.DeepMultiHeadMlpPolicy",
                    "policy used")
flags.DEFINE_string("policy_config", "", "config used for policy")
flags.DEFINE_string("type", "PPO", "PPO|PPO2|Vtrace actor type")
flags.DEFINE_string("self_infserver_addr", "", "infserver_addr self_agent used.")
flags.DEFINE_string("distill_infserver_addr", "", "infserver_addr distill_agent used.")
flags.DEFINE_string("post_process_data", None,
                    "post process of (X, A), drop useless mask in SC2.")
flags.DEFINE_boolean("compress", True, "whether data is compressed for infserver")
flags.DEFINE_boolean("rwd_shape", True, "do reward shape in actor")
flags.DEFINE_boolean("distillation", False, "use distillation policy")
flags.DEFINE_string("distill_policy_config", "", "config used for distill policy")
# printing/logging
flags.DEFINE_integer("verbose", 11,
                     "verbosity. The smaller, the noisier. Reference:"
                     "10:DEBUG, 20:INFO, 30: WARN, 40: ERROR, 50:DISABLED")
flags.DEFINE_integer("log_interval_steps", 51,
                     "frequency of printing log in steps")
flags.DEFINE_string("replay_dir", "", "replay dir when available")
#
flags.DEFINE_boolean(
  "reboot_on_failure", False,
  "Should actor reboot on failure. NOTE: Before rebooting, it kills ALL the "
  "Children SC2 processes on the machine. Use it carefully. Some hints:"
  "For k8s, we don't need rebooting as k8s can do it. "
)


def main(_):
  if FLAGS.replay_dir:
    os.makedirs(FLAGS.replay_dir, exist_ok=True)

  env_config = read_config_dict(FLAGS.env_config)
  interface_config = read_config_dict(FLAGS.interface_config)
  env = create_env(FLAGS.env, env_config=env_config,
                   inter_config=interface_config)
  policy = import_module_or_data(FLAGS.policy)
  policy_config = read_config_dict(FLAGS.policy_config)
  distill_policy_config = read_config_dict(FLAGS.distill_policy_config)
  post_process_data = None
  if FLAGS.post_process_data is not None:
    post_process_data = import_module_or_data(FLAGS.post_process_data)
  if FLAGS.type == 'PPO':
    Actor = PPOActor
  elif FLAGS.type == 'PPO2':
    Actor = PPO2Actor
  elif FLAGS.type == 'VTrace':
    Actor = VtraceActor
  elif FLAGS.type == 'DDPG':
    Actor = DDPGActor
  else:
    raise KeyError(f'Not recognized learner type {FLAGS.type}!')
  actor = Actor(env, policy,
                policy_config=policy_config,
                league_mgr_addr=FLAGS.league_mgr_addr or None,
                model_pool_addrs=FLAGS.model_pool_addrs.split(','),
                learner_addr=FLAGS.learner_addr,
                unroll_length=FLAGS.unroll_length,
                update_model_freq=FLAGS.update_model_freq,
                n_v=FLAGS.n_v,
                verbose=FLAGS.verbose,
                log_interval_steps=FLAGS.log_interval_steps,
                rwd_shape=FLAGS.rwd_shape,
                distillation=FLAGS.distillation,
                distill_policy_config=distill_policy_config,
                replay_dir=FLAGS.replay_dir,
                compress=FLAGS.compress,
                self_infserver_addr=FLAGS.self_infserver_addr or None,
                distill_infserver_addr=FLAGS.distill_infserver_addr or None,
                post_process_data=post_process_data)

  n_failures = 0
  while True:
    try:
      actor.run()
    except Exception as e:
      if not FLAGS.reboot_on_failure:
        raise e
      print("Actor crushed no. {}, the exception:\n{}".format(n_failures, e))
      n_failures += 1
      print("Rebooting...")
      kill_sc2_processes_v2()


if __name__ == '__main__':
  app.run(main)
