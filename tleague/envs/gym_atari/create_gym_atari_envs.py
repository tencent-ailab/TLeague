import gym
from gym.spaces import Tuple as GymTuple


class SingleAgentWrapper(gym.Wrapper):
  """Thin wrapper to make explicit a gym env as single agent.

  Always wrap a length one gym.spaces.Tuple for the environment passed in,
  where the `length one' means a single agent.
  """
  def __init__(self, env):
    super(SingleAgentWrapper, self).__init__(env)
    self.observation_space = GymTuple([env.observation_space])
    self.action_space = GymTuple([env.action_space])

  def reset(self, **kwargs):
    obs = self.env.reset()  # do not pass args to reset()
    return (obs,)

  def step(self, action):
    obs, rwd, done, info = super(SingleAgentWrapper, self).step(action[0])
    return (obs,), (rwd,), done, info


def create_gym_atari_env(arena_id):
  # TODO(pengsun): should refactor the Arena code
  from arena.wrappers.pong2p.pong2p_wrappers import ScaledFloatFrame

  env_name = arena_id.strip('gym_')
  env = gym.make(env_name)
  from gym.envs import atari
  if isinstance(env.env, atari.atari_env.AtariEnv):
    env = ScaledFloatFrame(env)
  env = SingleAgentWrapper(env)
  return env


def gym_atari_env_space(arena_id):
  env = create_gym_atari_env(arena_id)
  env.reset()
  ac_space = env.action_space.spaces[0]
  ob_space = env.observation_space.spaces[0]
  env.close()
  return ob_space, ac_space