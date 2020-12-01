from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
if sys.version_info.major > 2:
  xrange = range

from tleague.learners.pg_learner import PGLearner
from tleague.utils.data_structure import VtraceData


class VtraceLearner(PGLearner):
  """Learner for Vtrace."""
  def __init__(self, league_mgr_addr, model_pool_addrs, learner_ports, rm_size,
               batch_size, ob_space, ac_space, policy, gpu_id, **kwargs):
    super(VtraceLearner, self).__init__(
      league_mgr_addr, model_pool_addrs, learner_ports, rm_size, batch_size,
      ob_space, ac_space, policy, gpu_id, data_type=VtraceData, **kwargs)