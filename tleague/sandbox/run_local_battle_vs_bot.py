import importlib

import joblib
from absl import app
from absl import flags
from tleague.actors.agent import PGAgent
from tleague.envs.create_envs import create_env

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "sc2vsbot", "task env")
flags.DEFINE_string("policy1", "tpolicies.ppo.policies.DeepMultiHeadMlpPolicy", "policy used for agent 1")
flags.DEFINE_string("model1", "./model", "model file used for agent 1")
flags.DEFINE_integer("n_v", 17, "number of value heads")
flags.DEFINE_string("policy_config", "", "config used for policy")
flags.DEFINE_integer("difficulty", 3, "difficulty of a computer bot")
flags.DEFINE_integer("episodes", 1, "number of episodes")

def main(_):
  env = create_env(FLAGS.env, difficulty = FLAGS.difficulty)
  obs = env.reset()
  print(env.observation_space.spaces)
  policy_module, policy_name = FLAGS.policy1.rsplit(".", 1)
  policy1 = getattr(importlib.import_module(policy_module), policy_name)
  policies = [policy1]
  if FLAGS.policy_config:
    config_module, config_name = FLAGS.policy_config.rsplit(".", 1)
    policy_config = getattr(importlib.import_module(config_module), config_name)
  else:
    policy_config = {}
  agents = [PGAgent(policy, ob_sp, ac_sp, n_v=FLAGS.n_v, scope_name=name, policy_config=policy_config)
            for policy, ob_sp, ac_sp, name in
            zip(policies,
                env.observation_space.spaces,
                env.action_space.spaces,
                ['p1'])]
  model_file1 = FLAGS.model1
  model_0 = joblib.load(model_file1)
  agents[0].load_model(model_0.model)
  agents[0].reset(obs[0])

  episodes = FLAGS.episodes
  iter = 0
  sum_rwd = []
  while iter < episodes:
    while True:
      if hasattr(env, 'render') and FLAGS.env not in ['sc2', 'sc2full_formal', 'sc2vsbot_unit_rwd_no_micro']:
        env.render()
      act = [agent.step(ob) for agent, ob in zip(agents, obs)]
      obs, rwd, done, info = env.step(act)
      if done:
        print(rwd)
        sum_rwd.append(rwd[0,0])
        obs = env.reset()
        break
    iter += 1
  print(sum_rwd)


if __name__ == '__main__':
  app.run(main)
