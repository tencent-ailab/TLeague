from tleague.envs.vizdoom.create_vizdoom_envs import create_vizdoom_env
from tleague.utils.gym_space import assert_matched


def vizdoom_cig2017_track1_test(env_config=None):
  env_config = env_config or {
    'num_players': 3,
    'num_bots': 2,
    'episode_timeout': 21000,
    'is_window_visible': True,
    'train_mode': 'frag',
  }
  inter_config = {}
  max_steps = 5000

  env = create_vizdoom_env('vizdoom_cig2017_track1', env_config, inter_config)
  obs = env.reset()
  print(obs)
  assert_matched(env.observation_space, obs)
  for i in range(0, max_steps):
    act = env.action_space.sample()
    assert_matched(env.action_space, act)
    obs, rwd, done, info = env.step(act)
    [ob is None or assert_matched(sp, ob)
     for sp, ob in zip(env.observation_space.spaces, obs)]
    print('step {}, obs {}'.format(i, obs))
    if done:
      obs = env.reset()
      assert_matched(env.observation_space, obs)


if __name__ == '__main__':
  vizdoom_cig2017_track1_test({
    'num_players': 3,
    'num_bots': 2,
    'episode_timeout': 21000,
    'is_window_visible': True,
    'train_mode': 'frag',
  })
  vizdoom_cig2017_track1_test({
    'num_players': 2,
    'num_bots': 2,
    'episode_timeout': 21000,
    'is_window_visible': True,
    'train_mode': 'navi',
  })