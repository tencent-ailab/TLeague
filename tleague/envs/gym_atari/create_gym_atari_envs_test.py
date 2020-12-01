from tleague.envs.gym_atari.create_gym_atari_envs import create_gym_atari_env
from tleague.utils.gym_space import assert_matched


def gym_atari_test(arena_id=None):
  arena_id = arena_id or 'gym_atari_breakout-v4'
  env = create_gym_atari_env(arena_id)
  obs = env.reset()
  print(obs)
  assert_matched(env.observation_space, obs)
  for i in range(5000):
    act = env.action_space.sample()
    assert_matched(env.action_space, act)
    obs, rwd, done, info = env.step(act)
    env.render()
    assert_matched(env.observation_space, obs)
    print('step: {}, rwd: {}, done: {}'.format(i, rwd, done))
    if done:
      obs = env.reset()
      assert_matched(env.observation_space, obs)

  env.close()


if __name__ == '__main__':
  gym_atari_test('gym_atari_breakout-v4')
  gym_atari_test('gym_atari_seaquest-v4')
  gym_atari_test('gym_atari_spaceinvaders-v4')