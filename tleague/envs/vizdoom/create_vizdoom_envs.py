from os import path

import arena
from gym.spaces import Box, Discrete


def _create_cig2107_track1_env(env_config, inter_config):
  from arena.env.vizdoom_env import VizdoomMPEnv, VizdoomVecRwd
  from arena.interfaces.raw_int import RawInt
  from arena.env.env_int_wrapper import EnvIntWrapper
  from arena.interfaces.vizdoom.obs_int import ReshapedFrameObsInt
  from arena.interfaces.vizdoom.act_int import Discrete6ActionInt

  env_config = env_config or {}
  inter_config = inter_config or {}
  num_players = (8 if 'num_players' not in env_config
                 else env_config['num_players'])
  num_bots = (0 if 'num_bots' not in env_config
              else env_config['num_bots'])
  episode_timeout = (21000 if 'episode_timeout' not in env_config
                     else env_config['episode_timeout'])
  is_window_visible = (False if 'is_window_visible' not in env_config
                       else env_config['is_window_visible'])
  train_mode = ('frag' if 'train_mode' not in env_config
                else env_config['train_mode'])  # {'frag' | 'navi'}
  env = VizdoomMPEnv(
    config_path=path.join(path.dirname(arena.__file__),
                          'utils/vizdoom/_scenarios/cig.cfg'),
    num_players=num_players,
    num_bots=num_bots,
    episode_timeout=episode_timeout,
    is_window_visible=is_window_visible,
    train_mode=train_mode
  )
  env = VizdoomVecRwd(env)

  def _install_interfaces(i_agent):
    inter = RawInt()
    inter = ReshapedFrameObsInt(inter, env.envs[i_agent])
    inter = Discrete6ActionInt(inter)
    return inter

  env = EnvIntWrapper(env, [_install_interfaces(i) for i in range(num_players)])
  return env


def create_vizdoom_env(arena_id, env_config=None, inter_config=None):
  if arena_id == 'vizdoom_dogfight':
    raise NotImplementedError
  elif arena_id == 'vizdoom_cig2017_track1':
    return _create_cig2107_track1_env(env_config, inter_config)
  elif arena_id == 'vizdoom_cig2017_track2':
    raise NotImplementedError
  else:
    raise ValueError('unknown vizdoom arena_id {}'.format(arena_id))


def vizdoom_env_space(arena_id, env_config=None, inter_config=None):
  env_config = env_config or {}
  inter_config = inter_config or {}
  if arena_id == 'vizdoom_dogfight':
    raise NotImplementedError
  elif arena_id == 'vizdoom_cig2017_track1':
    ob_space = Box(shape=(168, 168, 6), low=0.0, high=255.)
    ac_space = Discrete(6)
  elif arena_id == 'vizdoom_cig2017_track2':
    raise NotImplementedError
  else:
    env = create_vizdoom_env(arena_id, env_config, inter_config)
    env.reset()
    ac_space = env.action_space.spaces[0]
    ob_space = env.observation_space.spaces[0]
    env.close()
  return ob_space, ac_space