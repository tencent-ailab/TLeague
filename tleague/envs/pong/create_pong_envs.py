def create_pong_env(arena_id):
  from arena.env.pong2p_env import Pong2pEnv
  from arena.wrappers.pong2p.pong2p_wrappers import ClipRewardEnv
  from arena.wrappers.pong2p.pong2p_wrappers import WarpFrame
  from arena.wrappers.pong2p.pong2p_wrappers import ScaledFloatFrame
  from arena.wrappers.pong2p.pong2p_wrappers import FrameStack
  from arena.wrappers.pong2p.pong2p_compete import NoRwdResetEnv

  assert arena_id in ['pong_2p']
  env = Pong2pEnv()
  env = WarpFrame(env)
  env = ClipRewardEnv(env)
  env = FrameStack(env, 4)
  env = ScaledFloatFrame(env)
  env = NoRwdResetEnv(env, no_reward_thres=1000)

  return env


def pong_env_space(arena_id):
  env = create_pong_env(arena_id)
  env.reset()
  ac_space = env.action_space.spaces[0]
  ob_space = env.observation_space.spaces[0]
  # TODO(pengsun): something wrong with .close()
  #env.close()
  return ob_space, ac_space