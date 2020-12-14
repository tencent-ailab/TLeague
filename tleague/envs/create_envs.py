from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app


RECOGNIZED_SC2_ID = {
  'sc2full_formal8_dict',
}

RECOGNIZED_POMMERMAN_ID = {
  'pommerman_v1',
  'pommerman_v1_fog',
  'pommerman_v2',
  'pommerman_v2_fog',
}

RECOGNIZED_VIZDOOM_ID = {
  'vizdoom_dogfight',
  'vizdoom_cig2017_track1',
  'vizdoom_cig2017_track2',
}

RECOGNIZED_SOCCER_ID = {
  'soccer',
}

RECOGNIZED_PONG_ID = {
  'pong_2p'
}


def create_env(arena_id, env_config=None, inter_config=None):
  """ create env from arena/env id using LAZY IMPORT, i.e., the corresponding
  game core (StarCraftII, Pommerman, ViZDoom,... ) is loaded only when used, and
   you don't have to install the game core when not used. """
  if arena_id in RECOGNIZED_SC2_ID:
    from tleague.envs.sc2 import create_sc2_env
    return create_sc2_env(arena_id, env_config=env_config,
                          inter_config=inter_config)
  elif arena_id in RECOGNIZED_POMMERMAN_ID:
    from tleague.envs.pomme import create_pommerman_env
    return create_pommerman_env(arena_id, env_config=env_config,
                                inter_config=inter_config)
  elif arena_id in RECOGNIZED_VIZDOOM_ID:
    from tleague.envs.vizdoom import create_vizdoom_env
    return create_vizdoom_env(arena_id, env_config, inter_config)
  elif arena_id in RECOGNIZED_SOCCER_ID:
    from tleague.envs.soccer import create_soccer_env
    return create_soccer_env(arena_id)
  elif arena_id in RECOGNIZED_PONG_ID:
    from tleague.envs.pong import create_pong_env
    return create_pong_env(arena_id)
  elif arena_id.startswith('gym_'):
    from tleague.envs.gym_atari import create_gym_atari_env
    return create_gym_atari_env(arena_id)
  else:
    raise Exception('Unknown arena_id {}'.format(arena_id))


def env_space(arena_id, env_config=None, inter_config=None):
  """ get observation_space and action_space from arena/env id.

  This is theoretically equivalent to querying the env.observation_space &
  env.action_space. However, for some env these two fields are only correctly
  set AFTER env.reset() is called (e.g., some SC2 env relies on the loaded map),
  which means the game core will be loaded and it can be time consuming.

  In this case, this env_space function can be helpful. It allows the caller to
  quickly get the spaces by "hacking" the arena/env id (when available) without
  having to install the game core. """
  if arena_id in RECOGNIZED_SC2_ID:
    from tleague.envs.sc2 import sc2_env_space
    return sc2_env_space(arena_id, env_config, inter_config)
  elif arena_id in RECOGNIZED_POMMERMAN_ID:
    from tleague.envs.pomme import pommerman_env_space
    return pommerman_env_space(arena_id, env_config, inter_config)
  elif arena_id in RECOGNIZED_VIZDOOM_ID:
    from tleague.envs.vizdoom import vizdoom_env_space
    return vizdoom_env_space(arena_id)
  elif arena_id in RECOGNIZED_SOCCER_ID:
    from tleague.envs.soccer import soccer_env_space
    return soccer_env_space(arena_id)
  elif arena_id in RECOGNIZED_PONG_ID:
    from tleague.envs.pong import pong_env_space
    return pong_env_space(arena_id)
  elif arena_id.startswith('gym_'):
    from tleague.envs.gym_atari import gym_atari_env_space
    return gym_atari_env_space(arena_id)
  else:
    raise Exception('Unknown arena_id {}'.format(arena_id))


def env_test(env_ids):
  # for temporary testing
  for env_id in env_ids:
    env = create_env(env_id)
    obs = env.reset()
    import time
    for i in range(10):
      env.render()
      time.sleep(0.1)
      act = env.action_space.sample()
      obs, rwd, done, info = env.step(act)
      if done:
        obs = env.reset()
    env.close()

def sc2_test(_):
  # for temporary testing
  zstat_dir = '/root/replay_ds/rp2124-mv8-victory-selected-Misc'
  zstat_dir = '/Users/jcxiong/SC2/rp1522-mmr-ge6000-winner-selected-8'

  interface_config = {
    'zstat_data_src': zstat_dir,
    'mmr': 7000,
    'max_bo_count': 20,
    'max_bobt_count': 20,
    'add_cargo_to_units': True,
    'correct_pos_radius': 3.5,
    'correct_building_pos': True,
    'crop_to_playable_area': False,
    'inj_larv_rule': True,
    'mof_lair_rule': True,
    'ban_zb_rule': False,
    'ban_rr_rule': False,
    'hydra_spire_rule': True,
    'overseer_rule': True,
    'expl_map_rule': False,
    'baneling_rule': True,
    'zmaker_version': 'v5',
    'verbose': 0}
  env_config = {
    'use_trt': False,
    'skip_noop': True,
    'early_term': False,
    'astar_rwd_version': 'v2',
    'compensate_win': False,
    'game_core_config': {
      'show_burrowed_shadows': True,
      'show_placeholders': False,
    },
  }

  env = create_env(arena_id='sc2full_formal8_dict', env_config=env_config,
                   inter_config=interface_config)
  import numpy as np
  for i in range(2):
    obs = env.reset()
    print(f'episode: {i}')
    start_locations = [ob.game_info.start_raw.start_locations[0] for ob in
                       env.unwrapped._obs]
    base_pos = [u.pos for ob in env.unwrapped._obs for u in
                ob.observation.raw_data.units if
                u.alliance == 1 and u.unit_type == 86]
    print('---------------')
    print(f'start_locations: {start_locations}')
    print(f'base_pos: {base_pos}')
    print('===============', flush=True)
    assert start_locations[0].x == base_pos[1].x and start_locations[1].x == \
           base_pos[0].x
    for _ in range(10):
      act = env.action_space.spaces[0].sample()
      act['A_SELECT'] = np.array([600] * 64, dtype=np.int32)
      act['A_CMD_UNIT'] = 0
      action = [act] * 2
      env.step(action)
  env.close()


if __name__ == '__main__':
  # app.run(sc2_test)
  app.run(env_test, list(RECOGNIZED_PONG_ID) + list(RECOGNIZED_POMMERMAN_ID))