import copy
from os import path
import pickle

from tleague.hyperparam_mgr import hyperparam_types
from tleague.hyperparam_mgr import configs
from tleague.utils import read_config_dict


class HyperparamMgr(object):
  """ Hyperparameter Manager """

  class Blackboard(object):
    """Blackboard for hyperparams.

    Storing shared information among hyperparam instances.
    The fields (self.__dict__) can be arbitrarily added/access by a specific
    hyperparam."""
    pass

  def __init__(self, model_pool_apis, mutable_hyperparam_type,
               hyperparam_config_name):
    self._model_pool_apis = model_pool_apis
    self._mutable_hyperparam_cls = getattr(hyperparam_types,
                                           mutable_hyperparam_type)
    if hyperparam_config_name is not None:
      try:
        self._hyperparam_config = getattr(configs, hyperparam_config_name)
      except:
        self._hyperparam_config = read_config_dict(hyperparam_config_name)
    else:
      self._hyperparam_config = {}

    self.blackboard = HyperparamMgr.Blackboard()

  def save(self, checkpoint_dir):
    filepath = path.join(checkpoint_dir, 'hyperparam_mgr')
    with open(filepath, 'wb') as f:
      pickle.dump(self.blackboard, f)

  def load(self, checkpoint_dir):
    filepath = path.join(checkpoint_dir, 'hyperparam_mgr')
    with open(filepath, 'rb') as f:
      self.blackboard = pickle.load(f)

  def get_learner_hyperparam(self, learner_id, copy_from_model_key,
                             is_mutate=True):
    if copy_from_model_key is None:
      # default hyper param
      hyperparam = self._default_hyperparam(learner_id)
      return copy.deepcopy(hyperparam)
    else:
      # hyperparam inherits (and optionally mutates) from copy_from_model_key
      hyperparam = self._model_pool_apis.pull_attr(
          'hyperparam', copy_from_model_key)
      if hyperparam is None:  # Imitation model without hyperparam
        new_hyperparam = self._default_hyperparam(learner_id)
      else:
        new_hyperparam = copy.deepcopy(hyperparam)
        if is_mutate:
          new_hyperparam.mutate(copy_from_model_key=copy_from_model_key)
      return copy.deepcopy(new_hyperparam)

  def _default_hyperparam(self, learner_id):
    kwargs = copy.deepcopy(self._hyperparam_config)
    kwargs['learner_id'] = learner_id
    kwargs['blackboard'] = self.blackboard
    return self._mutable_hyperparam_cls(**kwargs)
