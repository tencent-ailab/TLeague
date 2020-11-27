from tleague.envs.pong.create_pong_envs import create_pong_env
from tleague.utils.gym_space import assert_matched


def pong_2p_test():
  env = create_pong_env('pong_2p')
  obs = env.reset()
  print(obs)
  assert_matched(env.observation_space, obs)
  for i in range(5000):
    act = env.action_space.sample()
    assert_matched(env.action_space, act)
    obs, rwd, done, info = env.step(act)
    assert_matched(env.observation_space, obs)
    print('step: {}, rwd: {}, done: {}'.format(i, rwd, done))
    if done:
      obs = env.reset()
      assert_matched(env.observation_space, obs)


if __name__ == '__main__':
  pong_2p_test()