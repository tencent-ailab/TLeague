class LeagueMgrMsg(object):
  def __init__(self, attr=None, key1=None, key2=None):
    self.attr = attr
    self.key1 = key1
    self.key2 = key2

class LeagueMgrErroMsg(object):
  def __init__(self, msg):
    self.msg = msg

class LeagueMgrOKMsg(object):
  def __init__(self, msg):
    self.msg = msg