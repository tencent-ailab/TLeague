from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ActorTask(object):

  def __init__(self, model_key1, model_key2, hyperparam):
    self.model_key1 = model_key1
    self.model_key2 = model_key2
    self.hyperparam = hyperparam

  def __str__(self):
    return str({'model_key1': self.model_key1,
                'model_key2': self.model_key2,
                'hyperparam': str(self.hyperparam)})


class LearnerTask(object):

  def __init__(self, model_key, hyperparam, parent_model_key=None):
    self.model_key = model_key
    self.parent_model_key = parent_model_key
    self.hyperparam = hyperparam

  def __str__(self):
    return str({'model_key': self.model_key,
                'parent_model_key': self.parent_model_key,
                'hyperparam': str(self.hyperparam),})


class MatchResult(object):

  def __init__(self, model_key1, model_key2, outcome, info={}):
    self.model_key1 = model_key1
    self.model_key2 = model_key2
    self.outcome = outcome
    self.info = info

  def __str__(self):
    return str({'model_key1': self.model_key1, 'model_key2': self.model_key2,
                'outcome': self.outcome, 'info': self.info})


def is_inherit(model_key1, model_key2):
  return model_key1.split(':')[1] == model_key2.split(':')[0]


class ImitationTask(object):
  def __init__(self, replay_name='', player_id=1, game_version='4.7.1', validation=False, **kwargs):
    if not replay_name.endswith(".SC2Replay"):
      replay_name = str(replay_name) + ".SC2Replay"
    self.replay_name = str(replay_name)
    self.player_id = int(player_id)
    self.game_version = str(game_version)
    if self.game_version.count(".") > 2:
      self.game_version = '.'.join(self.game_version.split('.')[:3])
    self.validation = bool(int(validation))

  def __str__(self):
    return 'Imitation Task: (replay_name: {}, player_id: {}, game_version: {}, validation: {})'.format(
      self.replay_name, self.player_id, self.game_version, self.validation
    )