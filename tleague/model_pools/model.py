from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time


class Model(object):

  def __init__(self, model, hyperparam, key, createtime=None,
               freezetime=None, updatetime=None):
    self.model = model
    self.hyperparam = hyperparam
    self.key = key
    self.createtime = createtime
    self.freezetime = freezetime
    self.updatetime = updatetime
    if self.updatetime is None:
      self.updatetime = time.strftime('%Y%m%d%H%M%S')

  def freeze(self):
    assert self.freezetime is None
    self.freezetime = self.updatetime

  def is_freezed(self):
    return self.freezetime is not None

  def __str__(self):
    return 'Model: (key: {}, createtime: {}, updatetime: {}, freezetime: {})'.format(
      self.key, self.createtime, self.updatetime, self.freezetime
    )
