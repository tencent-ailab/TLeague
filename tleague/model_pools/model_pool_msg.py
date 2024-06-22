
class ModelPoolWriterMsg(object):
  def __init__(self, model=None, learner_meta=None, freeze=None):
    self.model = model
    self.learner_meta = learner_meta
    self.freeze = freeze


class ModelPoolReaderMsg(object):
  def __init__(self, attr=None, key=None):
    self.attr = attr
    self.key = key


class ModelPoolErroMsg(object):

  def __init__(self, msg):
    self.msg = msg
