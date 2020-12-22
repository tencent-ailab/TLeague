import joblib
import time
import numpy as np
from absl import app
from absl import flags
from tleague.envs.create_envs import create_env, env_space
from tleague.actors.agent import PGAgent
from tleague.utils import read_config_dict
from tleague.utils import import_module_or_data


FLAGS = flags.FLAGS
flags.DEFINE_string("interface_config", "{}", "interface_config")
flags.DEFINE_string("env_id", "pommerman_v2_fog", "env_config")
flags.DEFINE_string("env_config", "{'random_side': True, \
                'agent_list': 'docker::multiagentlearning/navocado,' \
                              'docker::multiagentlearning/navocado,' \
                              'docker::multiagentlearning/navocado,' \
                              'docker::multiagentlearning/navocado', \
                'rule_agents': [1, 3]}", "env_config")
flags.DEFINE_string("policy", "tpolicies.net_zoo.pommerman.conv_lstm", "policy used for agent")
flags.DEFINE_string("model", "./model", "model file used for agent")
flags.DEFINE_string("replay_dir", None, "replay path to store")
flags.DEFINE_string("policy_config", "", "config used for policy")
flags.DEFINE_integer("episodes", 1, "number of episodes")
flags.DEFINE_boolean("render", False, "whether to render")


def main(_):
  inter_config = read_config_dict(FLAGS.interface_config)
  env_config = read_config_dict(FLAGS.env_config)
  env_config['replay_dir'] = FLAGS.replay_dir
  env = create_env(FLAGS.env_id, env_config=env_config, inter_config=inter_config)

  # policy_config = {
  #   'use_xla': False,
  #   'rollout_len': 1,
  #   'test': True,
  #   'rl': False,
  #   'use_loss_type': 'none',
  #   'use_value_head': False,
  #   'use_self_fed_heads': True,
  #   'use_lstm': True,
  #   'nlstm': 64,
  #   'hs_len': 128,
  #   'lstm_duration': 1,
  #   'lstm_dropout_rate': 0.0,
  #   'lstm_cell_type': 'lstm',
  #   'lstm_layer_norm': True,
  #   'weight_decay': 0.00000002,
  #   'n_v': 11,
  #   'merge_pi': False,
  # }
  policy = import_module_or_data(FLAGS.policy)
  policy_config = read_config_dict(FLAGS.policy_config)
  n_v = policy_config['n_v'] if 'n_v' in policy_config else 1
  model_path = FLAGS.model
  model = joblib.load(model_path)
  obs = env.reset()
  print(env.observation_space)
  agent = PGAgent(policy, env.observation_space.spaces[0],
                  env.action_space.spaces[0], n_v,
                  policy_config=policy_config, scope_name='model')
  agent.load_model(model.model)

  for _ in range(FLAGS.episodes):
    agent.reset(obs[0])
    sum_rwd = 0
    while True:
      if FLAGS.render:
        env.render()
        time.sleep(0.1)
      act = [agent.step(obs[0]), [0, 0]]
      obs, rwd, done, info = env.step(act)
      sum_rwd += np.array(rwd)
      if done:
        if FLAGS.render:
          env.render()
          time.sleep(1)
        print(f'reward sum: {sum_rwd}, info: {info}')
        obs = env.reset()
        break
  print('--------------------------------')
  env.close()


if __name__ == '__main__':
  app.run(main)
