import os
from collections import OrderedDict

import numpy as np
from gym import RewardWrapper, Wrapper
from gym import spaces
from pysc2.env import sc2_env
from arena.env.sc2_base_env import SC2BaseEnv
from arena.interfaces.interface import Interface
from arena.env.env_int_wrapper import EnvIntWrapper, SC2EnvIntWrapper
from arena.wrappers.sc2_wrapper import VecRwd
from arena.wrappers.sc2_wrapper import EarlyTerminate
from arena.utils.spaces import NoneSpace
from arena.wrappers.sc2stat_wrapper import StatAllAction
from arena.wrappers.sc2stat_wrapper import StatZStatFn
from arena.wrappers.basic_env_wrapper import OppoObsAsObs


AGENT_INTERFACE = 'feature'
VISUALIZE = False
SCORE_INDEX = -1
# TODO(pengsun): these settings should be set by parsing arena_id
DEFAULT_MAP_NAME = 'AbyssalReef' # 'AbyssalReef' | 'KairosJunction'
DISABLE_FOG = False
SCORE_MULTIPLIER = None
MAX_RESET_NUM = 20
MAX_STEPS_PER_EPISODE = 48000
SCREEN_RESOLUTION = 64

RACES = {
    "R": sc2_env.Race.random,
    "P": sc2_env.Race.protoss,
    "T": sc2_env.Race.terran,
    "Z": sc2_env.Race.zerg,
}
DIFFICULTIES = {
    1: sc2_env.Difficulty.very_easy,
    2: sc2_env.Difficulty.easy,
    3: sc2_env.Difficulty.medium,
    4: sc2_env.Difficulty.medium_hard,
    5: sc2_env.Difficulty.hard,
    6: sc2_env.Difficulty.harder,
    7: sc2_env.Difficulty.very_hard,
    8: sc2_env.Difficulty.cheat_vision,
    9: sc2_env.Difficulty.cheat_money,
    10: sc2_env.Difficulty.cheat_insane,
}


class SeqLevDist_with_Coord(object):
  ## Compute LevDist (modified) using sequential added (item, pos) pair ##
  def __init__(self, target_order, target_boc):
    self.len = int(np.sum(target_order))
    self.target_order = target_order[:self.len]
    self.target_boc = target_boc[:self.len]
    self.dist = np.arange(self.len + 1)
    self.index = 0

  @staticmethod
  def binary2int(bx):
    x = 0
    for i in bx:
      x = 2*x + i
    return x

  def pos_dist(self, pos1, pos2):
    MAX_DIST = 6 # 2 gateways
    max_d = 0.8 # scaled to [0, max_d]
    assert len(pos1) == len(pos2)
    LEN = int(len(pos1)/2)
    if len(pos1) > 2: # binary encoding
      pos1 = [self.binary2int(pos1[:LEN]) / 2,
              self.binary2int(pos1[LEN:]) / 2, ]
      pos2 = [self.binary2int(pos2[:LEN]) / 2,
              self.binary2int(pos2[LEN:]) / 2, ]
    sq_dist = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    return max_d * min(1, sq_dist / MAX_DIST)

  def lev_dist(self, order, boc):
    index_now = int(np.sum(order))
    assert index_now >= self.index
    for index in range(self.index+1, index_now+1):
      new_dist = index * np.ones(self.len + 1)
      for i in range(1, self.len + 1):
        if all(order[index-1] == self.target_order[i-1]):
          new_dist[i] = self.dist[i - 1] + self.pos_dist(boc[index-1], self.target_boc[i-1])
        else:
          new_dist[i] = self.dist[i - 1] + 1
        new_dist[i] = min(new_dist[i-1]+1, self.dist[i]+1, new_dist[i])
      self.dist = new_dist
    self.index = index_now
    return self.dist[-1]

  def min_dist_possible(self):
    add_on = np.abs(np.arange(self.len + 1) - self.index)
    return np.min(self.dist + add_on)


class SC2ZergZStatVecReward(RewardWrapper):
  """ Reward shaping for zstat. Works for player 0, and fill
  arbitrary value for player 1. Output vectorized reward. """
  def __init__(self, env, dict_space=True, version='v2', compensate_win=True):
    super(SC2ZergZStatVecReward, self).__init__(env)
    self.dict_space = dict_space
    self.version = version
    self.compensate_win = compensate_win
    self.rwd_dim = 4 + (version == 'v3')

  def reset(self, **kwargs):
    obs = super(SC2ZergZStatVecReward, self).reset(**kwargs)
    self._potential_func = self._create_potential_func(obs[0], self.dict_space, self.version)
    self._last_potential = self._potential_func(self.unwrapped._obs[0], obs, False)
    return obs

  @staticmethod
  def _create_potential_func(obs, dict_space, version):
    if version in ['v2', 'v3']:
      from timitate.lib6.zstat_utils import BUILD_ORDER_OBJECT_CANDIDATES, RESEARCH_ABILITY_CANDIDATES
      split_indices = [len(BUILD_ORDER_OBJECT_CANDIDATES), -len(RESEARCH_ABILITY_CANDIDATES)]
    def _get_zstat(ob, prefix='Z_'):
      order_bt, boc_bt = None, None
      if dict_space:
        uc = np.split(ob[prefix + 'UNIT_COUNT'] > 0, split_indices)
        order = ob[prefix + 'BUILD_ORDER']
        boc = ob[prefix + 'BUILD_ORDER_COORD']
        if version == 'v3':
          order_bt = ob[prefix + 'BUILD_ORDER_BT']
          boc_bt = ob[prefix + 'BUILD_ORDER_COORD_BT']
      else:
        uc = np.split(ob[15] > 0, split_indices)
        order = ob[16]
        boc = ob[17]
        if version == 'v3':
          order_bt = ob[18]
          boc_bt = ob[19]
      return uc, order, boc, order_bt, boc_bt
    target_uc, target_order, target_boc, target_order_bt, target_boc_bt = _get_zstat(obs, 'Z_')
    SeqLevDist = SeqLevDist_with_Coord(target_order, target_boc)
    if version == 'v3':
      SeqLevDist_bt = SeqLevDist_with_Coord(target_order_bt, target_boc_bt)
    use_zstat = np.sum(target_order) > 0
    def potential_func(raw_pb, ob, win):
      if use_zstat:
        uc, order, boc, order_bt, boc_bt = _get_zstat(obs, 'IMM_')
        d_order = [- SeqLevDist.lev_dist(order, boc) / float(target_order.shape[0])]
        # negative Hamming Distance as potential for binary unit count
        d_uc = [ - sum(u != target_u) / float(len(target_u)) for u, target_u in zip(uc, target_uc)]
        if win:
          d_order = [- SeqLevDist.min_dist_possible() / float(target_order.shape[0])]
          d_uc = [- sum(u & ~target_u) / float(len(target_u)) for u, target_u in zip(uc, target_uc)]
        d_order_bt = []
        if version == 'v3':
          d_order_bt = [- SeqLevDist_bt.lev_dist(order_bt, boc_bt) / float(target_order_bt.shape[0])]
          if win:
            d_order_bt = [- SeqLevDist_bt.min_dist_possible() / float(target_order_bt.shape[0])]
        return d_order + d_order_bt + d_uc
      else:
        return [0] * (4 + (version == 'v3'))
    return potential_func

  def step(self, actions):
    obs, rwd, done, info = self.env.step(actions)
    win = done and (rwd[0] > 0) and self.compensate_win
    potential = self._potential_func(self.unwrapped._obs[0], obs, win)
    game_loop = self.unwrapped._obs[0].observation.game_loop
    ratio = np.ones([self.rwd_dim])
    if game_loop > 22.4 * 60 * 8:
      ratio[-4:] *= 0.5
    if game_loop > 22.4 * 60 * 16:
      ratio[-4:] *= 0.5
    if game_loop > 22.4 * 60 * 24:
      ratio[-4:] = 0
    r = [(p-lp) * rr for p, lp, rr in zip(potential, self._last_potential, ratio)]
    self._last_potential = potential
    rwd[0] = list(rwd[0]) + r if isinstance(rwd[0], (list, tuple)) else [rwd[0]] + r
    if len(rwd) == 2:
      rwd[1] = list(rwd[1]) + [0] * self.rwd_dim if isinstance(rwd[1], (list, tuple))\
        else [rwd[1]] + [0] * self.rwd_dim
    return obs, rwd, done, info


class OppoObsComponentsV2:
  def __init__(self, game_version, dict_space, max_bo_count,
               max_bobt_count, zmaker_version, **kwargs):
    from timitate.lib6.pb2feature_converter import GlobalFeatMaker,  \
      ImmZStatMaker
    self._dict_space = dict_space
    self._components = [GlobalFeatMaker(game_version),
                        ImmZStatMaker(max_bo_count=max_bo_count,
                                      max_bobt_count=max_bobt_count,
                                      zstat_version=zmaker_version)]

  def make(self, pb):
    if self._dict_space:
      dict_features = OrderedDict()
      for com in self._components:
        dict_features.update(
          OrderedDict(['OPPO_'+k, v] for k, v in zip(com.tensor_names, com.make(pb))))
      return dict_features
    else:
      tuple_features = ()
      for com in self._components:
        tuple_features += com.make(pb)

  @property
  def space(self):
    if self._dict_space:
      items = []
      for com in self._components:
        items += zip(com.tensor_names, com.space.spaces)
      return spaces.Dict(OrderedDict(['OPPO_'+k, v] for k, v in items))
    else:
      items = ()
      for com in self._components:
        items += com.space.spaces
      return spaces.Tuple(list(items))


class SC2ZergOppoObsAsObs(OppoObsAsObs):
  """ SC2Zerg specific parse func V1; may design other versions in future """
  def __init__(self, env, dict_space, attachment):
    super(SC2ZergOppoObsAsObs, self).__init__(env)
    self._attachment = attachment
    self._dict_space = dict_space
    self._expand_obs_space()

  def _expand_obs_space(self):
    space_old = self.observation_space.spaces[self._me_id]
    if self._dict_space:
      assert isinstance(space_old, spaces.Dict)
      self.observation_space.spaces[self._me_id] = \
          spaces.Dict(OrderedDict(list(space_old.spaces.items()) +
                                  list(self._attachment.space.spaces.items())))
    else:
      assert not isinstance(space_old, spaces.Dict)
      if isinstance(self._attachment.space, spaces.Tuple):
          self.observation_space.spaces[self._me_id] = \
              spaces.Tuple(tuple(space_old.spaces) + tuple(self._attachment.space.spaces))
      else:
          self.observation_space.spaces[self._me_id] = \
              spaces.Tuple(tuple(space_old.spaces) + (self._attachment.space,))

  def _parse_oppo_obs(self, raw_oppo_obs):
    # raw_oppo_obs should be a timestep
    pb = raw_oppo_obs, None
    return self._attachment.make(pb)


class StatZStatDist(Wrapper):
  """Statistics for ZStat Distance between target and imm"""
  def __init__(self, env):
    super(StatZStatDist, self).__init__(env)
    self.SeqLevDist = {}
    self.info = {}
    self.required_keys = ['Z_BUILD_ORDER', 'Z_BUILD_ORDER_COORD',
                          'IMM_BUILD_ORDER', 'IMM_BUILD_ORDER_COORD']

  def reset(self, **kwargs):
    self.SeqLevDist = {}
    self.info = {}
    return super(StatZStatDist, self).reset(**kwargs)

  def step(self, actions):
    obs, reward, done, info = self.env.step(actions)
    for ind, ob in enumerate(obs):
      if ob is not None:
        key = f'agt{ind}-zstat-dist'
        if isinstance(ob, dict) and set(self.required_keys) <= set(ob):
          if np.sum(ob['Z_BUILD_ORDER']) > 0:
            if ind not in self.SeqLevDist:
              self.SeqLevDist[ind] = SeqLevDist_with_Coord(
                ob['Z_BUILD_ORDER'], ob['Z_BUILD_ORDER_COORD'])
            self.info[key] = self.SeqLevDist[ind].lev_dist(
              ob['IMM_BUILD_ORDER'], ob['IMM_BUILD_ORDER_COORD'])
        else:
          print(f"WARN: cannot find the fields {self.required_keys} for agent{ind}")
    for k in self.info:
      info[k] = self.info[k]
    return obs, reward, done, info


# TODO(pengsun): strict base env creation and interface creation and remove
#  those create_sc2*_env functions
def make_sc2_base_env(n_players=2, step_mul=8, version='4.7.0',
                      screen_resolution=SCREEN_RESOLUTION, screen_ratio=1.33,
                      camera_width_world_units=24, map_name=DEFAULT_MAP_NAME,
                      replay_dir=None, use_pysc2_feature=True,
                      game_core_config={}):
  players = [sc2_env.Agent(sc2_env.Race.zerg) for _ in range(n_players)]
  if replay_dir:
    os.makedirs(replay_dir, exist_ok=True)
  save_replay_episodes = 1 if replay_dir else 0
  return SC2BaseEnv(
    players=players,
    agent_interface=AGENT_INTERFACE,
    map_name=map_name,
    screen_resolution=screen_resolution,
    screen_ratio=screen_ratio,
    camera_width_world_units=camera_width_world_units,
    visualize=VISUALIZE,
    step_mul=step_mul,
    disable_fog=DISABLE_FOG,
    score_index=SCORE_INDEX,
    score_multiplier=SCORE_MULTIPLIER,
    max_reset_num=MAX_RESET_NUM,
    max_steps_per_episode=MAX_STEPS_PER_EPISODE,
    version=version,
    replay_dir=replay_dir,
    save_replay_episodes=save_replay_episodes,
    use_pysc2_feature=use_pysc2_feature,
    minimap_resolution=(camera_width_world_units, screen_resolution),
    game_core_config=game_core_config
  )


def make_sc2full_v8_interface(zstat_data_src='',
                              mmr=3500,
                              dict_space=False,
                              zstat_presort_order_name=None,
                              zmaker_version='v4',
                              output_map_size=(128, 128),
                              **kwargs):
  from arena.interfaces.sc2full_formal.obs_int import FullObsIntV7
  from arena.interfaces.sc2full_formal.act_int import FullActIntV6, NoopActIntV4
  from arena.interfaces.raw_int import RawInt
  from arena.interfaces.sc2full_formal.obs_int import ActAsObsSC2
  noop_nums = [i+1 for i in range(128)]
  inter = RawInt()
  # this obs inter requires game core 4.10.0
  inter = FullObsIntV7(inter, zstat_data_src=zstat_data_src,
                       mmr=mmr,
                       dict_space=dict_space,
                       zstat_presort_order_name=zstat_presort_order_name,
                       game_version='4.10.0',
                       zmaker_version=zmaker_version,
                       output_map_resolution=output_map_size,
                       **kwargs)
  inter = FullActIntV6(inter, max_noop_num=len(noop_nums),
                       map_resolution=output_map_size,
                       dict_space=dict_space,
                       **kwargs)
  inter = ActAsObsSC2(inter)
  noop_func = lambda x: x['A_NOOP_NUM'] if dict_space else x[1]
  inter = NoopActIntV4(inter, noop_nums=noop_nums, noop_func=noop_func)
  return inter


def create_sc2full_formal_env(inter_fun,
                              inter_config=None,
                              vec_rwd=True,
                              unit_rwd=True,
                              astar_rwd=False,
                              astar_rwd_version='v1',
                              compensate_win=True,
                              early_term=True,
                              stat_action=True,
                              stat_zstat_fn=True,
                              version='4.7.0',
                              map_name=DEFAULT_MAP_NAME,
                              replay_dir=None,
                              step_mul=None,
                              centralized_value=False,
                              use_trt=False,
                              stat_zstat_dist=True,
                              skip_noop=False,
                              game_core_config={}
                              ):
  if inter_fun in [make_sc2full_v8_interface]:
    # default to 1
    step_mul = 1 if step_mul is None else step_mul
    # TODO(pengsun): double-check the screen_xxx stuff
    env = make_sc2_base_env(n_players=2, step_mul=step_mul, version=version,
                            screen_resolution=168, screen_ratio=0.905,
                            camera_width_world_units=152, map_name=map_name,
                            replay_dir=replay_dir, use_pysc2_feature=False,
                            game_core_config=game_core_config,
                            )
  else:
    raise ValueError('Unknown interface fun {}'.format(inter_fun))
  if early_term:
    env = EarlyTerminate(env)
  if stat_action:
    env = StatAllAction(env)
  if vec_rwd:
    env = VecRwd(env, append=True)
  if use_trt:
    # install Tower-Rush-Trick env wrapper
    from arena.wrappers.sc2_wrapper import OppoTRTNoOut
    env = OppoTRTNoOut(env)

  # parse interface config
  inter_config = {} if inter_config is None else inter_config
  # really install the interfaces
  inter1 = inter_fun(**inter_config)
  inter2 = inter_fun(**inter_config)
  if skip_noop and inter_fun == make_sc2full_v8_interface:
    noop_func = lambda x: x['A_NOOP_NUM'] + 1
    env = SC2EnvIntWrapper(env, [inter1, inter2], noop_func)
  else:
    env = EnvIntWrapper(env, [inter1, inter2])

  # Note: add it AFTER EnvIntWrapper
  if stat_zstat_fn:
    env = StatZStatFn(env)

  if stat_zstat_dist:
    env = StatZStatDist(env)

  # install other possible env wrappers
  if astar_rwd:
    env = SC2ZergZStatVecReward(env, version=astar_rwd_version, compensate_win=compensate_win)
  if centralized_value:
    dict_space = False if 'dict_space' not in inter_config else inter_config['dict_space']
    if inter_fun in [make_sc2full_v8_interface]:
      zmaker_version = 'v4' if 'zmaker_version' not in inter_config else inter_config['zmaker_version']
      max_bo_count = 50 if 'max_bo_count' not in inter_config else inter_config['max_bo_count']
      max_bobt_count = 20 if 'max_bobt_count' not in inter_config else inter_config['max_bobt_count']
      env = SC2ZergOppoObsAsObs(env, dict_space, OppoObsComponentsV2(
        version, dict_space, max_bo_count, max_bobt_count, zmaker_version))
    else:
      raise NotImplemented('Unknown interface using centralized value.')
  return env


def create_sc2_env(arena_id, env_config=None, inter_config=None):
  # parsing env config
  env_config = {} if env_config is None else env_config
  difficulty = (0 if 'difficulty' not in env_config
                else env_config['difficulty'])
  replay_dir = (None if 'replay_dir' not in env_config
                else env_config['replay_dir'])
  step_mul = None if 'step_mul' not in env_config else env_config['step_mul']
  map_name = ('KairosJunction' if 'map_name' not in env_config
              else env_config['map_name'])
  use_trt = False if 'use_trt' not in env_config else env_config['use_trt']
  astar_rwd_version = ('v3' if 'astar_rwd_version' not in env_config
                       else env_config['astar_rwd_version'])
  compensate_win = (True if 'compensate_win' not in env_config
                    else env_config['compensate_win'])
  skip_noop = False if 'skip_noop' not in env_config else env_config['skip_noop']
  # By deault, game_core_config = {'crop_to_playable_area': False,
  #                                'show_cloaked': False,
  #                                'show_burrowed_shadows': False,
  #                                'show_placeholders': False,
  #                                'raw_affects_selection': True,}
  game_core_config = ({} if 'game_core_config' not in env_config
                      else env_config['game_core_config'])
  early_term = (True if 'early_term' not in env_config
                else env_config['early_term'])  # enabling by default
  if arena_id == 'sc2full_formal8_dict':
    inter_config = {} if inter_config is None else inter_config
    inter_config['dict_space'] = True
    return create_sc2full_formal_env(
      inter_fun=make_sc2full_v8_interface,
      inter_config=inter_config,
      vec_rwd=False,
      unit_rwd=False,
      astar_rwd=True,
      astar_rwd_version=astar_rwd_version,
      compensate_win=compensate_win,
      early_term=early_term,
      map_name=map_name,
      version='4.10.0',  # required game core version 4.10.0
      replay_dir=replay_dir,
      step_mul=1,  # enforce step_mul=1 TODO(pengsun): double-check this
      centralized_value=True,
      use_trt=use_trt,
      skip_noop=skip_noop,
      game_core_config=game_core_config,
    )
  else:
    raise Exception('Unknown arena_id {}'.format(arena_id))


def sc2_env_space(arena_id, env_config=None, inter_config=None):
  # parsing env config
  env_config = {} if env_config is None else env_config
  inter_config = {} if inter_config is None else inter_config
  if 'max_bo_count' not in inter_config:
    inter_config['max_bo_count'] = 50
  if 'max_bobt_count' not in inter_config:
    inter_config['max_bobt_count'] = 20
  if 'zmaker_version' not in inter_config:
    inter_config['zmaker_version'] = 'v5'
  centralized_value = True if 'centralized_value' not in env_config else env_config['centralized_value']
  import distutils
  version = '4.10.0'
  version_ge_414 = (distutils.version.LooseVersion(version)
                    >= distutils.version.LooseVersion('4.1.4'))
  if arena_id in ['sc2full_formal8_dict']:
    # TODO(pengsun): need interface_config here?? now it's hard-coding
    inter_config['dict_space'] = True
    inter = make_sc2full_v8_interface(
      **inter_config
    )
    ob_space = inter.observation_space
    ac_space = inter.action_space
    # TODO(pengsun): need interface_config here?? now it's hard-coding
    if centralized_value:
      com = OppoObsComponentsV2(game_version='4.10.0',
                                lib='lib6',
                                **inter_config)
      ob_space = spaces.Dict(OrderedDict(list(ob_space.spaces.items()) +
                                         list(com.space.spaces.items())))
  else:
    env = create_sc2_env(arena_id, env_config, inter_config)
    env.reset()
    ac_space = env.action_space.spaces[0]
    ob_space = env.observation_space.spaces[0]
    env.close()
  return ob_space, ac_space


def filter_mask(obs, act):
  if isinstance(obs, spaces.Dict) and isinstance(act, spaces.Dict):
    for key in ['MASK_SELECTION', 'MASK_CMD_UNIT', 'MASK_CMD_POS']:
      obs.spaces[key].shape = obs.spaces[key].shape[1:]
  elif obs is not None:
    for key in ['MASK_SELECTION', 'MASK_CMD_UNIT', 'MASK_CMD_POS']:
      obs[key] = obs[key][act['A_AB']]
  return obs, act
