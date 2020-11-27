from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from arena.env.pommerman_env import pommerman_2v2
from arena.env.pommerman_env import vec_rwd
from arena.env.env_int_wrapper import EnvIntWrapper
from arena.interfaces.common import ActAsObs, MultiBinObsInt, \
  FrameStackInt, BoxTransformInt
from arena.wrappers.basic_env_wrapper import AllObs
from arena.interfaces.combine import Combine


def mk_pommerman_int_v1(**kwargs):
  # two observations are combined and merged.
  from arena.interfaces.pommerman.obs_int import BoardMapObs, AttrObsInt, \
    PosObsInt, ActMaskObsInt, CombineObsInt
  inter = CombineObsInt(None)
  inter = BoardMapObs(inter)
  inter = FrameStackInt(inter, 4)
  op = lambda x: np.reshape(np.transpose(x, (1, 2, 0, 3)), (11, 11, -1))
  inter = BoxTransformInt(inter, op, index=0)
  inter = ActAsObs(inter, override=False, n_action=10)
  inter = MultiBinObsInt(inter, lambda obs: obs['step_count'], override=False)
  inter = AttrObsInt(inter, override=False)
  inter = PosObsInt(inter, override=False)
  inter = ActMaskObsInt(inter)
  return inter


def mk_pommerman_int_v2(**kwargs):
  from arena.interfaces.pommerman.obs_int_v2 import BoardMapObs, \
    AttrObsInt, ActMaskObsInt, RotateInt, PosObsInt
  # inter1 = RotateInt(None)
  inter1 = BoardMapObs(None, use_attr=True, override=True)
  inter1 = PosObsInt(inter1, override=False)
  inter1 = ActMaskObsInt(inter1, override=False)

  # inter2 = RotateInt(None)
  inter2 = BoardMapObs(None, use_attr=True, override=True)
  inter2 = PosObsInt(inter2, override=False)
  inter2 = ActMaskObsInt(inter2, override=False)

  inter = Combine(None, [inter1, inter2])
  return inter


def create_pommerman_env(arena_id, env_config=None, inter_config=None):
  env_config = {} if env_config is None else env_config
  random_side = (True if 'random_side' not in env_config
                 else env_config['random_side'])
  centralV = (False if 'centralV' not in env_config
              else env_config['centralV'])
  rotate = (False if 'rotate' not in env_config
              else env_config['rotate'])
  agent_list = (None if 'agent_list' not in env_config
              else env_config['agent_list'])
  rule_agents = ([] if 'rule_agents' not in env_config
              else env_config['rule_agents'])
  replay_dir = (None if 'replay_dir' not in env_config
                else env_config['replay_dir'])
  if arena_id in ['pommerman_v1', 'pommerman_v2']:
    env = pommerman_2v2('PommeTeam-v0', random_side=random_side,
                        agent_list=agent_list, rule_agents=rule_agents,
                        replay_dir=replay_dir)
  elif arena_id in ['pommerman_v1_fog', 'pommerman_v2_fog']:
    env = pommerman_2v2('PommeTeamCompetition-v0', random_side=random_side,
                        agent_list=agent_list, rule_agents=rule_agents,
                        replay_dir=replay_dir)
  else:
    raise Exception('Unknown arena_id {}'.format(arena_id))
  if rotate:
    from arena.interfaces.pommerman.obs_int_v2 import RotateInt
    env = EnvIntWrapper(env, [RotateInt(None), RotateInt(None)])
  env = vec_rwd(env)
  inter_fun = None
  if arena_id in ['pommerman_v1', 'pommerman_v1_fog']:
    inter_fun = mk_pommerman_int_v1
  elif arena_id in ['pommerman_v2', 'pommerman_v2_fog']:
    inter_fun = mk_pommerman_int_v2
  assert inter_fun
  inter_config = {} if inter_config is None else inter_config
  inter1 = inter_fun(**inter_config)
  inter2 = inter_fun(**inter_config)
  env = EnvIntWrapper(env, [inter1, inter2])
  if centralV:
    env = AllObs(env)
  return env


def pommerman_env_space(arena_id, env_config=None, inter_config=None):
  env = create_pommerman_env(arena_id, env_config=env_config,
                             inter_config=inter_config)
  env.reset()
  ac_space = env.action_space.spaces[0]
  ob_space = env.observation_space.spaces[0]
  env.close()
  return ob_space, ac_space
