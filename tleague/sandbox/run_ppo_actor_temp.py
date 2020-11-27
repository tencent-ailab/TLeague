from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
from absl import app
from absl import flags
from tleague.envs.create_envs import create_env
from tleague.ppo_actor_temp import PPOActor

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
flags.DEFINE_string("policy", "tpolicies.ppo.policies.DeepMultiHeadMlpPolicy", "policy used")
flags.DEFINE_boolean("rwd_shape", True, "do reward shape in actor")
# printing/logging
flags.DEFINE_integer("verbose", 11,
                     "verbosity. The smaller, the noisier. Reference:"
                     "10:DEBUG, 20:INFO, 30: WARN, 40: ERROR, 50:DISABLED")
flags.DEFINE_integer("log_interval_steps", 51,
                     "frequency of printing log in steps")


def main(_):
  env = create_env(FLAGS.env)
  policy_module, policy_name = FLAGS.policy.rsplit(".", 1)
  policy = getattr(importlib.import_module(policy_module), policy_name)
  actor = PPOActor(env, policy,
                   league_mgr_addr=FLAGS.league_mgr_addr,
                   model_pool_addrs=FLAGS.model_pool_addrs.split(','),
                   learner_addr=FLAGS.learner_addr,
                   unroll_length=FLAGS.unroll_length,
                   update_model_freq=FLAGS.update_model_freq,
                   n_v=FLAGS.n_v,
                   verbose=FLAGS.verbose,
                   log_interval_steps=FLAGS.log_interval_steps,
                   rwd_shape=FLAGS.rwd_shape)
  actor.run()


if __name__ == '__main__':
  app.run(main)
