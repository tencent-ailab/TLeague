import joblib
from absl import app
from absl import flags
from pysc2.env import sc2_env
from arena.env.sc2_base_env import SC2BaseEnv
from tleague.actors.agent import PGAgent2
from tleague.utils import read_config_dict
from tleague.utils import import_module_or_data
from arena.env.env_int_wrapper import EnvIntWrapper


FLAGS = flags.FLAGS
flags.DEFINE_string("interface", "tleague.envs.sc2.create_sc2_envs.make_sc2full_v8_interface", "task env")
flags.DEFINE_string("interface_config", "", "interface_config")
flags.DEFINE_string("policy", "tpolicies.net_zoo.mnet_v6.mnet_v6d4", "policy used for agent")
flags.DEFINE_string("model", "./model", "model file used for agent")
flags.DEFINE_string("version", "4.10.0", "game core version")
flags.DEFINE_string("replay_dir", './', "replay path to store")
flags.DEFINE_integer("n_v", 17, "number of value heads")
flags.DEFINE_string("policy_config", "", "config used for policy")
flags.DEFINE_integer("difficulty", 7, "difficulty of a computer bot")
flags.DEFINE_integer("episodes", 1, "number of episodes")


def main(_):
  players = [sc2_env.Agent(sc2_env.Race.zerg),
             sc2_env.Bot(sc2_env.Race.zerg, FLAGS.difficulty)]
  env = SC2BaseEnv(
    players=players,
    agent_interface='feature',
    map_name='KairosJunction',
    max_steps_per_episode=48000,
    screen_resolution=168,
    screen_ratio=0.905,
    step_mul=1,
    version=FLAGS.version,
    replay_dir=FLAGS.replay_dir,
    save_replay_episodes=1,
    use_pysc2_feature=False,
    minimap_resolution=(152, 168),
  )
  interface_cls = import_module_or_data(FLAGS.interface)
  interface_config = read_config_dict(FLAGS.interface_config)
  interface = interface_cls(**interface_config)
  env = EnvIntWrapper(env, [interface])
  obs = env.reset()
  print(env.observation_space.spaces)
  policy = import_module_or_data(FLAGS.policy)
  policy_config = read_config_dict(FLAGS.policy_config)
  agent = PGAgent2(policy,
                   env.observation_space.spaces[0],
                   env.action_space.spaces[0],
                   policy_config=policy_config)
  model_path = FLAGS.model
  model = joblib.load(model_path)
  agent.load_model(model.model)
  agent.reset(obs[0])

  episodes = FLAGS.episodes
  iter = 0
  sum_rwd = []
  while True:
    while True:
      if obs[0] is not None:
        act = [agent.step(obs[0])]
      else:
        act = [[]]
      obs, rwd, done, info = env.step(act)
      if done:
        print(rwd)
        sum_rwd.append(rwd[0])
        break
    iter += 1
    if iter >= episodes:
      print(sum_rwd)
      break
    obs = env.reset()


if __name__ == '__main__':
  app.run(main)
