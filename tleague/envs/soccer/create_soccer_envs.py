from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from arena.env.soccer_env import soccer_gym
from arena.env.env_int_wrapper import EnvIntWrapper 
from arena.interfaces.combine import Combine
from arena.interfaces.soccer.obs_int import ConcatObsAct, Dict2Vec

def create_soccer_env(arena_id):
  team_size = 2
  env = soccer_gym(team_size)
  intf1 = Combine(None, [Dict2Vec(None), Dict2Vec(None)])
  intf1 = ConcatObsAct(intf1)
  intf2 = Combine(None, [Dict2Vec(None), Dict2Vec(None)])
  intf2 = ConcatObsAct(intf2)
  env = EnvIntWrapper(env, [intf1, intf2])
  return env

def soccer_env_space(arena_id):
  env = create_soccer_env(arena_id)
  env.reset()
  ac_space = env.action_space.spaces[0]
  ob_space = env.observation_space.spaces[0]
  env.close()
  return ob_space, ac_space