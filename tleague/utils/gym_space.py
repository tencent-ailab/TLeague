from collections import OrderedDict

import numpy as np
from gym.spaces import Discrete
from gym.spaces import MultiBinary
from gym.spaces import MultiDiscrete
from gym.spaces import Box
from gym.spaces import Tuple
from gym.spaces import Dict


def assert_matched(space, space_inst):
  """Assert a gym space must match its instance.

  Implemented as recursive assert.
  E.g., observation matches observation_space, action matches action_space, etc.

  Args:
    space: gym space (e.g., observation_space, action_space)
    space_inst: gym space instance (e.g., observation, action)
  """
  if isinstance(space, Dict):
    assert type(space_inst) == OrderedDict
    for sp_name, sp_inst_name in zip(space.spaces.keys(), space_inst.keys()):
      assert sp_name == sp_inst_name
    for sp, sp_inst in zip(space.spaces.values(), space_inst.values()):
      assert_matched(sp, sp_inst)
  elif isinstance(space, Tuple):
    assert type(space_inst) == list or type(space_inst) == tuple
    assert len(space.spaces) == len(space_inst)
    for sp, sp_inst in zip(space.spaces, space_inst):
      assert_matched(sp, sp_inst)
  elif isinstance(space, Box):
    assert isinstance(space_inst, np.ndarray)
    assert space.shape == space_inst.shape
  elif isinstance(space, Discrete):
    print('skipped checking for Discrete')
    pass
  elif isinstance(space, MultiBinary):
    print('skipped checking for MultiBinary')
    pass
  elif isinstance(space, MultiDiscrete):
    print('skipped checking for MultiDiscrete')
    pass
  else:
    print('not impled checking for space {}, skipped'.format(space))


if __name__ == '__main__':
  space = Tuple([
    Box(0.0, 1.0, shape=(2, 3), dtype=np.float32),
    Dict([
      ('x', Discrete(6)),
      ('y', Box(0.0, 1.0, shape=(8, 32), dtype=np.float32))
    ])
  ])
  inst = (
    np.ones(shape=(2, 3), dtype=np.float32),
    OrderedDict([
      ('x', 5),
      ('y', np.zeros(shape=(8, 32), dtype=np.float32))
    ])
  )
  assert_matched(space, inst)
  assert_matched(space, space.sample())