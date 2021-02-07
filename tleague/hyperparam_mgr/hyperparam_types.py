import random
from random import randint

import numpy as np


class Hyperparam(object):
  def __init__(self, **kwargs):
    for k in kwargs:
      setattr(self, k, kwargs[k])

  def mutate(self, **kwargs):
    pass


class MutableHyperparam(Hyperparam):

  def __init__(self, learning_rate=1e-4, cliprange=0.1, lam=0.9, gamma=0.99,
               reward_weights=[1.0]*11, reward_weights_disturb=0.1, **kwargs):
    self.learning_rate = learning_rate
    self.cliprange = cliprange
    self.lam = lam
    self.gamma = gamma
    self.reward_weights = reward_weights
    self.reward_weights_disturb = reward_weights_disturb
    super().__init__(**kwargs)

  def mutate(self, **kwargs):
    for i in range(len(self.reward_weights)):
      delta = random.choice([-1.0, 1.0]) * self.reward_weights_disturb
      self.reward_weights[i] += delta

  def __str__(self):
    s_nn = _str_dict({'lr': self.learning_rate, 'cliprange': self.cliprange,
                      'lam': self.lam, 'gamma': self.gamma})
    s_weight = 'rwd weight: ' + _str_list(self.reward_weights)
    return type(self).__name__ + ': ' + ', '.join([s_nn, s_weight])


def _str_dict(kv):
  return ", ".join(["{}: {}".format(k, v) for k, v in kv.items()])


def _str_list(ell):
  return ", ".join(["{}: {}".format(i, item) for i, item in enumerate(ell)])


def _loguniform(low=0.0, high=1.0):
  """ generate a random NUMBER(SCALAR) using LogUniform distribution """
  verysmall = 1e-8
  return np.exp(np.random.uniform(np.log(low + verysmall), np.log(high), None))


class MutableHyperparamRandPredefSC2V1(Hyperparam):
  """ Random Weight drawn from a set of predefined weights, V1. For SC2 only.

  Expect dim=11 vectorized unit production reward in the order:
  game_output,
  zergling, baneling, queen, roach, ravager,
  infestor, hydralisk, mutalisk, broodlord, corruptor

  Each choice corresponds to:
  * NO_PREFERENCE
  * ling + baneling
  * roach + infestor
  * roach + hydralisk
  * ling + baneling + mutalisk
  * roach + ravager
  * mutualisk + broodlord + corrupter
  """
  def __init__(self, learning_rate=1e-4, cliprange=0.1, lam=0.9, gamma=0.99,
               sigma=1/6.0, reward_weights=[1.0] * 11, **kwargs):
    # for RL training
    self.learning_rate = learning_rate
    self.cliprange = cliprange
    self.lam = lam
    self.gamma = gamma
    # for PBT
    self.sigma = sigma

    initial_ind = 2
    u = 0.005
    hu = u/2
    self.available_weights = [
      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [1, u, u, 0, 0, 0, 0, 0, 0, 0, 0],
      [1, 0, 0, 0, u, 0, u, 0, 0, 0, 0],
      [1, 0, 0, 0, u, 0, 0, u, 0, 0, 0],
      [1, hu, hu, 0, 0, 0, 0, 0, u, 0, 0],
      [1, 0, 0, 0, u, u, 0, 0, 0, 0, 0],
      [1, 0, 0, 0, 0, 0, 0, 0, hu, u, u],
    ]
    self.reward_weights = self.available_weights[initial_ind]
    super().__init__(**kwargs)

  def mutate(self, **kwargs):
    self.reward_weights = random.choice(self.available_weights)

  def __str__(self):
    s_nn = _str_dict({
      'lr': self.learning_rate, 'cliprange': self.cliprange, 'lam': self.lam,
      'gamma': self.gamma, 'sigma': self.sigma
    })
    s_weight = 'rwd weight: ' + _str_list(self.reward_weights)
    return type(self).__name__ + ': ' + ', '.join([s_nn, s_weight])


class MutableHyperparamRandPredefSC2V2(Hyperparam):
  """ Random Weight drawn from a set of predefined weights, V2. For SC2 only.

  Expect dim=11 vectorized unit production reward in the order:
  game_output,
  zergling, baneling, queen, roach, ravager,
  infestor, hydralisk, mutalisk, broodlord, corruptor

  Each choice corresponds to:
  * NO_PREFERENCE
  * ling + baneling
  * roach + infestor
  * roach + hydralisk
  * ling + baneling + mutalisk
  * roach + ravager
  * mutualisk + broodlord + corrupter
  """
  def __init__(self, learning_rate=1e-4, cliprange=0.1, lam=0.9, gamma=0.99,
               available_sigma=(1/6.0, 1/60.0), initial_sigma=1/6.0,
               reward_scale=0.005, perturb_prob=0.2, **kwargs):
    # for RL training
    self.learning_rate = learning_rate
    self.cliprange = cliprange
    self.lam = lam
    self.gamma = gamma
    # for PBT
    self.sigma = initial_sigma
    # for Reward
    self.reward_weights = None  # will be set later

    self._available_sigma = available_sigma
    self._perturb_prob = perturb_prob

    self._reward_scale = reward_scale
    u = self._reward_scale
    hu = u/2
    self._available_weights = [
      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [1, u, u, 0, 0, 0, 0, 0, 0, 0, 0],
      [1, 0, 0, 0, u, 0, u, 0, 0, 0, 0],
      [1, 0, 0, 0, u, 0, 0, u, 0, 0, 0],
      [1, hu, hu, 0, 0, 0, 0, 0, u, 0, 0],
      [1, 0, 0, 0, u, u, 0, 0, 0, 0, 0],
      [1, 0, 0, 0, 0, 0, 0, 0, hu, u, u],
    ]
    initial_ind = 1
    self.reward_weights = self._available_weights[initial_ind]
    super().__init__(**kwargs)

  def mutate(self, **kwargs):
    if random.random() < self._perturb_prob:
      self.reward_weights = random.choice(self._available_weights)
      self.sigma = random.choice(self._available_sigma)

  def __str__(self):
    s_nn = _str_dict({
      'lr': self.learning_rate, 'cliprange': self.cliprange, 'lam': self.lam,
      'gamma': self.gamma, 'sigma': self.sigma
    })
    s_weight = 'rwd weight: ' + _str_list(self.reward_weights)
    return type(self).__name__ + ': ' + ', '.join([s_nn, s_weight])


class MutableHyperparamRandMaxKOnes(Hyperparam):
  """ Random Weight. At most K ones, others are zeros. Entry 0 always be 1.0

  For a vectorized reward, rwd, with length 1+N, it firstly generates a random
  integer M (1 <= M <= K), then it generates a weight vector with M ones at
  random entries (from 1 to N) and zeros otherwise.
  weight[0] always = 1.0 as it is assumed that rwd[0] indicates game outcome.
  """
  def __init__(self, learning_rate=1e-4, cliprange=0.1, lam=0.9, gamma=0.99,
               reward_weights=[1.0] * 11, K=4, **kwargs):
    self.learning_rate = learning_rate
    self.cliprange = cliprange
    self.lam = lam
    self.gamma = gamma

    self.reward_len = len(reward_weights)
    assert(self.reward_len > 1)
    self.K = min(K, self.reward_len - 1)
    self.reward_weights = [0.0] * self.reward_len
    super().__init__(**kwargs)

  def mutate(self, **kwargs):
    u = 0.005
    M = random.randint(1, self.K)
    ones_ind = random.sample(range(self.reward_len - 1), M)
    tmp = [0.0 for _ in range(self.reward_len - 1)]
    for i in ones_ind:
      tmp[i] = u
    self.reward_weights = [1.0] + tmp

  def __str__(self):
    s_nn = _str_dict({'lr': self.learning_rate, 'cliprange': self.cliprange,
                      'lam': self.lam, 'gamma': self.gamma})
    s_weight = 'rwd weight: ' + _str_list(self.reward_weights)
    return type(self).__name__ + ': ' + ', '.join([s_nn, s_weight])


class MutableHyperparamRandPerturb(Hyperparam):
  """ Random initial weights and random perturbation when mutating.

  Implement the mutating algorithm described in Sect 5.5 & Sect 2.2 of the
  paper:

  Max Jaderberg et el. Human-level performance in first-person multiplayer games
   with population-based deep reinforcement learning.
  """
  def __init__(self, learning_rate=1e-4, cliprange=0.1, lam=0.9, gamma=0.99,
               available_sigma=(1/6.0, 1/60.0), initial_sigma=1/6.0,
               reward_len = 11, weight_perturb_percentage=0.2,
               perturb_prob=0.2, init_weight_low=0.1, init_weight_high=10.0,
               lockup_weights_ind=(0,), reward_scale=0.05, **kwargs):
    # for RL training
    self.learning_rate = learning_rate
    self.cliprange = cliprange
    self.lam = lam
    self.gamma = gamma
    # for PBT
    self.sigma = initial_sigma
    # for Reward
    self.reward_weights = None  # will be set later

    self._available_sigma = available_sigma
    self._lockup_weights_ind = lockup_weights_ind
    self._perturb_prob = perturb_prob
    self._weight_perturb_percentage = weight_perturb_percentage

    # (re-)generate initial reward weights
    # TODO(pengsun): allow +1/-1 sign as in the paper?
    self._reward_len = reward_len
    if isinstance(reward_scale, list) or isinstance(reward_scale, tuple):
      assert len(reward_scale) == self._reward_len
    else:
      reward_scale = [reward_scale] * self._reward_len
    self.reward_weights = [
      _loguniform(init_weight_low, init_weight_high) * scale
      for scale in reward_scale
    ]
    self._lockup_weights_one()
    super().__init__(**kwargs)

  def mutate(self, **kwargs):
    if random.random() < self._perturb_prob:
      # perturb reward weights by a -/+ percentage
      ratios = [np.random.uniform(1 - self._weight_perturb_percentage,
                                  1 + self._weight_perturb_percentage)
                for _ in self.reward_weights]
      self.reward_weights = [w * r for w, r in zip(self.reward_weights, ratios)]
      self._lockup_weights_one()
      # perturb sigma by random selection
      self.sigma = random.choice(self._available_sigma)

  def _lockup_weights_one(self):
    """ lockup some weight(s) to absolute value 1.0 """
    for i in self._lockup_weights_ind:
      self.reward_weights[i] = 1.0

  def __str__(self):
    s_nn = _str_dict({
      'lr': self.learning_rate, 'cliprange': self.cliprange, 'lam': self.lam,
      'gamma': self.gamma, 'sigma': self.sigma
    })
    s_weight = 'rwd weight: ' + _str_list(self.reward_weights)
    s_lockup = 'lockup weight ind: ' + _str_list(self._lockup_weights_ind)
    return type(self).__name__ + ': ' + ', '.join([s_nn, s_weight, s_lockup])


class MutableHyperparamRandPredefSigma(Hyperparam):
  """ Random sigma from pre-def sets. Sigma is used when computing Elo matching
  score during PBT training. Do the mutation with a probability.

  Implement the algorithm described in the following papers:
  [1] Max Jaderberg et el. Human-level performance in first-person multiplayer
  games with population-based deep reinforcement learning.
  [2] Emergent Coordination Through Competition, ICLR2019
  [3] AlphaStar Blog.
  """
  def __init__(self, learning_rate=1e-4, cliprange=0.1, lam=0.9, gamma=0.99,
               available_sigma=(1/6.0, 1/60.0), initial_sigma=1/6.0,
               reward_weights=(1.0,), perturb_prob=0.2, **kwargs):
    # for RL training
    self.learning_rate = learning_rate
    self.cliprange = cliprange
    self.lam = lam
    self.gamma = gamma
    # for PBT
    self.sigma = initial_sigma
    # for reward
    self.reward_weights = reward_weights

    self._available_sigma = available_sigma
    self._perturb_prob = perturb_prob
    super().__init__(**kwargs)

  def mutate(self, **kwargs):
    if random.random() < self._perturb_prob:
      # perturb sigma by random selection
      self.sigma = random.choice(self._available_sigma)

  def __str__(self):
    s_nn = _str_dict({
      'lr': self.learning_rate, 'cliprange': self.cliprange, 'lam': self.lam,
      'gamma': self.gamma, 'sigma': self.sigma
    })
    s_weight = 'rwd weight: ' + _str_list(self.reward_weights)
    return type(self).__name__ + ': ' + ', '.join([s_nn, s_weight])


class MutableHyperparamPartialPerturb(Hyperparam):
  """ Random initial weights and random perturbation when mutating; Support
  ndarray weight; Support sigma used by PBT.
  """
  def __init__(self, learning_rate=1e-4, cliprange=0.1, lam=0.9, gamma=0.99,
               available_sigma=(1/6.0, 1/60.0), initial_sigma=1/6.0,
               weight_perturb_percentage=0.2, initial_reward_weights=[1],
               perturb_prob=1.0, init_weight_low=0.1, init_weight_high=10.0,
               perturb_ind=(0,), reward_scale=0.1, **kwargs):
    # for RL training
    self.learning_rate = learning_rate
    self.cliprange = cliprange
    self.lam = lam
    self.gamma = gamma
    # for PBT
    self.sigma = initial_sigma
    # for Reward
    self.reward_weights = np.array(initial_reward_weights)
    assert len(self.reward_weights.shape) > 0

    self._available_sigma = available_sigma
    self._perturb_ind = perturb_ind
    self._perturb_prob = perturb_prob
    self._weight_perturb_percentage = weight_perturb_percentage

    # generate initial reward weights with partial initializer
    self._perturb_len = len(perturb_ind)
    if isinstance(reward_scale, list) or isinstance(reward_scale, tuple):
      assert len(reward_scale) == self._perturb_len
    else:
      reward_scale = [reward_scale] * self._perturb_len
    for ind, scale in zip(self._perturb_ind, reward_scale):
      self.reward_weights[ind] = scale * _loguniform(init_weight_low, init_weight_high)
    super().__init__(**kwargs)

  def mutate(self, **kwargs):
    if random.random() < self._perturb_prob:
      # perturb reward weights by a -/+ percentage
      for ind in self._perturb_ind:
        ratio = np.random.uniform(1 - self._weight_perturb_percentage,
                                  1 + self._weight_perturb_percentage)
        self.reward_weights[ind] = self.reward_weights[ind] * ratio
      # perturb sigma by random selection
      self.sigma = random.choice(self._available_sigma)

  def __str__(self):
    s_nn = _str_dict({
      'lr': self.learning_rate, 'cliprange': self.cliprange, 'lam': self.lam,
      'gamma': self.gamma, 'sigma': self.sigma
    })
    s_weight = 'rwd weight: ' + _str_list(self.reward_weights)
    return type(self).__name__ + ': ' + ', '.join([s_nn, s_weight])


class MutableHyperparamPreDefPartialPerturb(Hyperparam):
  """ Random initial weights from pre-defined and random perturbation when mutating;
   Support ndarray weight; Support sigma used by PBT.
  """
  def __init__(self, learning_rate=1e-4, cliprange=0.1, lam=0.9, gamma=0.99,
               available_sigma=(1/6.0, 1/60.0), initial_sigma=1/6.0,
               weight_perturb_percentage=0.2, initial_reward_weights=[[1,],[2,]],
               perturb_prob=1.0, perturb_ind=(0,), reward_scale=0.1, **kwargs):
    # for RL training
    self.learning_rate = learning_rate
    self.cliprange = cliprange
    self.lam = lam
    self.gamma = gamma
    # for PBT
    self.sigma = initial_sigma
    self._available_sigma = available_sigma
    self._perturb_ind = perturb_ind
    self._perturb_prob = perturb_prob
    self._weight_perturb_percentage = weight_perturb_percentage

    # for Reward
    self.reward_weights = np.array(random.choice(initial_reward_weights), dtype=np.float32)
    assert len(self.reward_weights.shape) > 0
    for ind in self._perturb_ind:
      self.reward_weights[ind] = self.reward_weights[ind] * reward_scale
    super().__init__(**kwargs)

  def mutate(self, **kwargs):
    if random.random() < self._perturb_prob:
      # perturb reward weights by a -/+ percentage
      for ind in self._perturb_ind:
        ratio = np.random.uniform(1 - self._weight_perturb_percentage,
                                  1 + self._weight_perturb_percentage)
        self.reward_weights[ind] = self.reward_weights[ind] * ratio
      # perturb sigma by random selection
      self.sigma = random.choice(self._available_sigma)

  def __str__(self):
    s_nn = _str_dict({
      'lr': self.learning_rate, 'cliprange': self.cliprange, 'lam': self.lam,
      'gamma': self.gamma, 'sigma': self.sigma
    })
    s_weight = 'rwd weight: ' + _str_list(self.reward_weights)
    return type(self).__name__ + ': ' + ', '.join([s_nn, s_weight])


class ConstantHyperparam(Hyperparam):
  """Constant Hyperparams"""
  def __init__(self, learning_rate=1e-4, cliprange=0.1, lam=0.9, gamma=0.99,
               reward_weights=None, sigma=1., **kwargs):
    self.learning_rate = learning_rate
    self.cliprange = cliprange
    self.lam = lam
    self.gamma = gamma
    self.reward_weights = reward_weights or [1.0]
    self.sigma = sigma
    super().__init__(**kwargs)

  def __str__(self):
    s_nn = _str_dict({'lr': self.learning_rate, 'cliprange': self.cliprange,
                      'lam': self.lam, 'gamma': self.gamma})
    s_weight = 'rwd weight: ' + _str_list(self.reward_weights)
    return type(self).__name__ + ': ' + ', '.join([s_nn, s_weight])


class DiscreteDistribHyperparam(ConstantHyperparam):
  """Discrete Distribution Hyperparameter

  Generates discrete distribution on init. Provides another zeroing_probability.
  Used for, e.g., the zstat file sampling for the Agent/Exploiter self-play.
  """

  def __init__(self, learner_id=None, n_distrib=100,
               lrn_id_to_distrib_type=None, zeroing_prob=0.2, **kwargs):
    self.n_distrib = n_distrib
    self.learner_id = learner_id or 'lrngrp0'
    self.lrn_id_to_distrib_type = lrn_id_to_distrib_type or {
      'lrngrp0': 'uniform', 'lrngrp1': 'one_hot', 'lrngrp2': 'one_hot'
    }
    self.zeroing_prob = zeroing_prob
    super(DiscreteDistribHyperparam, self).__init__(**kwargs)

    # generate the concrete distribution
    assert self.learner_id in self.lrn_id_to_distrib_type, \
      'learner_id: {}, lrn_id_to_distrib_type: {}'.format(
        self.learner_id, self.lrn_id_to_distrib_type)
    dt = self.lrn_id_to_distrib_type[self.learner_id]
    if dt == 'uniform':
      self.distrib = (np.ones(shape=(self.n_distrib,), dtype=np.float32)
                      / self.n_distrib)
    elif dt == 'one_hot':
      self.distrib = np.zeros(shape=(self.n_distrib,), dtype=np.float32)
      self.distrib[randint(0, self.n_distrib - 1)] = 1.0
    else:
      raise ValueError('Unknown dstrib type: {}'.format(dt))

  def __str__(self):
    s_nn = _str_dict({
      'lr': self.learning_rate,
      'cliprange': self.cliprange,
      'lam': self.lam,
      'gamma': self.gamma,
      'sigma': self.sigma,
    })
    s_distrib = ('n_distrib: {}, learner_id: {}, distrib_type: {}, '
                 'zeroing_prob: {}').format(
      self.n_distrib,
      self.learner_id,
      self.lrn_id_to_distrib_type[self.learner_id],
      self.zeroing_prob
    )
    s_weight = 'rwd weight: ' + _str_list(self.reward_weights)
    return type(self).__name__ + ': ' + ', '.join([s_nn, s_distrib, s_weight])


class DiscreteDistribHyperparamV2(ConstantHyperparam):
  """Discrete Distribution Hyperparameter, V2

  On init, generates the discrete distribution, the zeroing_probability, and the
  reward weights (as well as its activating prob) according to the learner id
  spec, which are then exposed through the following public members:
    distrib
    zero_prob
    rewrad_weights
  """
  def __init__(self,
               learner_id=None,
               blackboard=None,
               lrn_id_to_n_distrib=None,
               lrn_id_to_max_n_cyclic=None,
               lrn_id_to_distrib_type=None,
               lrn_id_to_zeroing_prob=None,
               lrn_id_to_reward_weights=None,
               lrn_id_to_reward_weights_activate_prob=None,
               lrn_id_to_init_model_key=None,
               lrn_id_to_distill_model_key=None,
               **kwargs):
    self.learner_id = learner_id or 'lrngrp0'
    self.lrn_id_to_init_model_key = lrn_id_to_init_model_key or {
      'lrngrp0': 'None:init_model0',
      'lrngrp1': 'None:init_model0',
      'lrngrp2': 'None:init_model0'
    }
    self.lrn_id_to_distill_model_key = lrn_id_to_distill_model_key or \
                                       self.lrn_id_to_init_model_key
    self.lrn_id_to_n_distrib = lrn_id_to_n_distrib or {
      'lrngrp0': 8,
      'lrngrp1': 8,
      'lrngrp2': 8
    }
    self.lrn_id_to_max_n_cyclic = lrn_id_to_max_n_cyclic or {
      # only for cyclic type lrngrp
      'lrngrp0': None,
      'lrngrp1': None,
      'lrngrp2': 4
    }
    self.lrn_id_to_distrib_type = lrn_id_to_distrib_type or {
      'lrngrp0': 'uniform',
      'lrngrp1': 'random_one_hot',
      'lrngrp2': 'cyclic_one_hot'
    }
    self.lrn_id_to_zeroing_prob = lrn_id_to_zeroing_prob or {
      'lrngrp0': 0.2, 'lrngrp1': 0.0, 'lrngrp2': 0.0
    }
    self.lrn_id_to_reward_weights = lrn_id_to_reward_weights or {
      'lrngrp0': [1.0] + [0.1] * 4,
      'lrngrp1': [1.0] + [0.0] * 4,
      'lrngrp2': [1.0] + [0.0] * 4
    }
    self.lrn_id_to_reward_weights_activate_prob = (
      lrn_id_to_reward_weights_activate_prob or {
      'lrngrp0': [1.0] + [0.25] * 4,
      'lrngrp1': [1.0] + [0.0] * 4,
      'lrngrp2': [1.0] + [0.0] * 4
    })
    super(DiscreteDistribHyperparamV2, self).__init__(**kwargs)

    # initialize blackboard.cur_idx_one_hot when necessary
    assert blackboard is not None
    if 'cur_idx_one_hot' not in blackboard.__dict__:
      print('DiscreteDistribHyperparamV2: Initialize'
            ' blackboard.cur_idx_one_hot only once')
      # a {lrn_id: count} dict for the distrib type 'cyclic_one_hot'
      blackboard.cur_idx_one_hot = {
        lrn_id: 0 for lrn_id in self.lrn_id_to_n_distrib
      }
    print('DiscreteDistribHyperparamV2: blackboard.cur_idx_one_hot={}'.format(
      blackboard.cur_idx_one_hot)
    )

    # get the n_distrib for this learner id
    self.n_distrib = self.lrn_id_to_n_distrib[self.learner_id]

    # generate the concrete distrib
    dt = self.lrn_id_to_distrib_type[self.learner_id]
    if dt == 'uniform':
      self.distrib = (np.ones(shape=(self.n_distrib,), dtype=np.float32)
                      / self.n_distrib)
    elif dt == 'random_one_hot':
      self.distrib = np.zeros(shape=(self.n_distrib,), dtype=np.float32)
      self.distrib[randint(0, self.n_distrib - 1)] = 1.0
    elif dt == 'cyclic_one_hot':
      # set the one hot at current entry
      self.distrib = np.zeros(shape=(self.n_distrib,), dtype=np.float32)
      self.distrib[blackboard.cur_idx_one_hot[self.learner_id]] = 1.0
      # move to the next entry
      blackboard.cur_idx_one_hot[self.learner_id] += 1
      assert self.lrn_id_to_max_n_cyclic[self.learner_id] is not None, (
        'None max_n_cyclic in {}'.format(self.learner_id)
      )
      blackboard.cur_idx_one_hot[self.learner_id] = (
        blackboard.cur_idx_one_hot[self.learner_id]
        % self.lrn_id_to_max_n_cyclic[self.learner_id]
      )
    else:
      raise ValueError('Unknown distrib type: {}'.format(dt))

    # generate the concrete zeroing_prob
    self.zeroing_prob = self.lrn_id_to_zeroing_prob[self.learner_id]

    # generate the concrete reward_weights
    # over-write the base class reward_weights
    self._reward_weights_orig = self.lrn_id_to_reward_weights[self.learner_id]
    self.reward_weights = self._mutate_reward_weights()
    self.init_model_key = self.lrn_id_to_init_model_key[self.learner_id]
    self.distill_model_key = self.lrn_id_to_distill_model_key[self.learner_id]

  def mutate(self, **kwargs):
    self.reward_weights = self._mutate_reward_weights()

  def _mutate_reward_weights(self):
    # check whether to zero it for each reward_weights element
    return [
      rw if random.random() < prob else 0.0  # activate it with activate_prob
      for rw, prob in zip(
        self._reward_weights_orig,
        self.lrn_id_to_reward_weights_activate_prob[self.learner_id]
      )
    ]

  def __str__(self):
    s_nn = _str_dict({
      'lr': self.learning_rate,
      'cliprange': self.cliprange,
      'lam': self.lam,
      'gamma': self.gamma,
      'sigma': self.sigma,
    })
    s_distrib = ('learner_id: {}, n_distrib: {}, distrib_type: {}, '
                 'distrib: {}, zeroing_prob: {}').format(
      self.learner_id,
      self.n_distrib,
      self.lrn_id_to_distrib_type[self.learner_id],
      self.distrib,
      self.zeroing_prob
    )
    s_weight = 'rwd_weights: ' + _str_list(self.reward_weights)
    s_weight_orig = 'rwd_weights_orig' + _str_list(self._reward_weights_orig)
    return type(self).__name__ + ': ' + ', '.join([s_nn, s_distrib, s_weight,
                                                   s_weight_orig])


class DiscreteDistribHyperparamV3(ConstantHyperparam):
  """Discrete Distribution Hyperparameter, V3

  On init, generates the zsta_category name, the zeroing_probability, and the
  reward weights (as well as its activating prob) according to the learner id
  spec, which are then exposed through the following public members:
    zstat_category
    zero_prob
    rewrad_weights
  """
  def __init__(self,
               learner_id=None,
               blackboard=None,
               lrn_id_to_zstat_category=None,
               lrn_id_to_zeroing_prob=None,
               lrn_id_to_reward_weights=None,
               lrn_id_to_reward_weights_activate_prob=None,
               lrn_id_to_init_model_key=None,
               lrn_id_to_distill_model_key=None,
               **kwargs):
    self.learner_id = learner_id or 'lrngrp0'
    self.lrn_id_to_init_model_key = lrn_id_to_init_model_key or {
      'lrngrp0': 'None:init_model0',
      'lrngrp1': 'None:init_model0',
      'lrngrp2': 'None:init_model0'
    }
    self.lrn_id_to_distill_model_key = lrn_id_to_distill_model_key or \
                                       self.lrn_id_to_init_model_key
    self.lrn_id_to_zstat_category = lrn_id_to_zstat_category or {
      'lrngrp0': 'Normal',
      'lrngrp1': 'PureRoach',
      'lrngrp2': 'Hydralisk'
    }
    self.lrn_id_to_zeroing_prob = lrn_id_to_zeroing_prob or {
      'lrngrp0': 0.2, 'lrngrp1': 0.0, 'lrngrp2': 0.0
    }
    self.lrn_id_to_reward_weights = lrn_id_to_reward_weights or {
      'lrngrp0': [1.0] + [0.1] * 4,
      'lrngrp1': [1.0] + [0.0] * 4,
      'lrngrp2': [1.0] + [0.0] * 4
    }
    self.lrn_id_to_reward_weights_activate_prob = (
      lrn_id_to_reward_weights_activate_prob or {
      'lrngrp0': [1.0] + [0.25] * 4,
      'lrngrp1': [1.0] + [0.0] * 4,
      'lrngrp2': [1.0] + [0.0] * 4
    })
    super(DiscreteDistribHyperparamV3, self).__init__(**kwargs)

    # generate the concrete zstat_category
    self.zstat_category = self.lrn_id_to_zstat_category[self.learner_id]

    # generate the concrete zeroing_prob
    self.zeroing_prob = self.lrn_id_to_zeroing_prob[self.learner_id]

    # generate the concrete reward_weights
    # over-write the base class reward_weights
    self._reward_weights_orig = self.lrn_id_to_reward_weights[self.learner_id]
    self.reward_weights = self._mutate_reward_weights()
    self.init_model_key = self.lrn_id_to_init_model_key[self.learner_id]
    self.distill_model_key = self.lrn_id_to_distill_model_key[self.learner_id]

  def mutate(self, copy_from_model_key=None, **kwargs):
    self.reward_weights = self._mutate_reward_weights()
    if copy_from_model_key is not None:
      self.distill_model_key = copy_from_model_key

  def _mutate_reward_weights(self):
    # check whether to zero it for each reward_weights element
    return [
      rw if random.random() < prob else 0.0  # activate it with activate_prob
      for rw, prob in zip(
        self._reward_weights_orig,
        self.lrn_id_to_reward_weights_activate_prob[self.learner_id]
      )
    ]

  def __str__(self):
    s_nn = _str_dict({
      'lr': self.learning_rate,
      'cliprange': self.cliprange,
      'lam': self.lam,
      'gamma': self.gamma,
      'sigma': self.sigma,
    })
    s_distrib = ('learner_id: {}, zstat_category: {}, '
                 'zeroing_prob: {}').format(
      self.learner_id,
      self.zstat_category,
      self.zeroing_prob
    )
    s_weight = 'rwd_weights: ' + _str_list(self.reward_weights)
    s_weight_orig = 'rwd_weights_orig' + _str_list(self._reward_weights_orig)
    return type(self).__name__ + ': ' + ', '.join([s_nn, s_distrib, s_weight,
                                                   s_weight_orig])


class LpLen(Hyperparam):
  max_total_timesteps = 0
  minimal_lp_len_ratio = 0.3
  default_burn_in_timesteps = 0
  need_burn_in = False

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.total_timesteps = int(self.max_total_timesteps *
                               (1 - (1 - self.minimal_lp_len_ratio) * random.random()))

  @property
  def burn_in_timesteps(self):
    if self.need_burn_in:
      self.need_burn_in = False
      return self.default_burn_in_timesteps
    else:
      return 0

  def mutate(self, **kwargs):
    super().mutate()
    self.total_timesteps = int(self.max_total_timesteps *
                               (1 - (1 - self.minimal_lp_len_ratio) * random.random()))
    self.need_burn_in = True

  def __str__(self):
    s_nn = _str_dict({
      'total_timesteps': self.total_timesteps, 'need_burn_in': self.need_burn_in,
      'default_burn_in_timesteps': self.default_burn_in_timesteps
    })
    return ', '.join([s_nn])


class LpLenMutableHyperparam(LpLen, MutableHyperparam):
  def __str__(self):
    return MutableHyperparam.__str__(self) + ', ' + LpLen.__str__(self)


class LpLenMutableHyperparamRandPredefSC2V1(LpLen, MutableHyperparamRandPredefSC2V1):
  def __str__(self):
    return MutableHyperparamRandPredefSC2V1.__str__(self) + ', ' + LpLen.__str__(self)


class LpLenMutableHyperparamRandPredefSC2V2(LpLen, MutableHyperparamRandPredefSC2V2):
  def __str__(self):
    return MutableHyperparamRandPredefSC2V2.__str__(self) + ', ' + LpLen.__str__(self)


class LpLenMutableHyperparamRandMaxKOnes(LpLen, MutableHyperparamRandMaxKOnes):
  def __str__(self):
    return MutableHyperparamRandMaxKOnes.__str__(self) + ', ' + LpLen.__str__(self)


class LpLenMutableHyperparamRandPerturb(LpLen, MutableHyperparamRandPerturb):
  def __str__(self):
    return MutableHyperparamRandPerturb.__str__(self) + ', ' + LpLen.__str__(self)


class LpLenMutableHyperparamRandPredefSigma(LpLen, MutableHyperparamRandPredefSigma):
  def __str__(self):
    return MutableHyperparamRandPredefSigma.__str__(self) + ', ' + LpLen.__str__(self)


class LpLenMutableHyperparamPartialPerturb(LpLen, MutableHyperparamPartialPerturb):
  def __str__(self):
    return MutableHyperparamPartialPerturb.__str__(self) + ', ' + LpLen.__str__(self)


class LpLenMutableHyperparamPreDefPartialPerturb(LpLen, MutableHyperparamPreDefPartialPerturb):
  def __str__(self):
    return MutableHyperparamPreDefPartialPerturb.__str__(self) + ', ' + LpLen.__str__(self)


class LpLenConstantHyperparam(LpLen, ConstantHyperparam):
  def __str__(self):
    return ConstantHyperparam.__str__(self) + ', ' + LpLen.__str__(self)
