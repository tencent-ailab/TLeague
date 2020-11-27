#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 20190312 by qing
# 20190404 modified by jc
# 20190412 contaminated by pengsun -_-

""" Game Managers with Matching Strategy

For detailed discussions on the algorithm, please refer to the following works:

[1] AlphaStar: Mastering the Real-Time Strategy Game StarCraft II
    https://deepmind.com/blog/alphastar-mastering-real-time-strategy-game-starcraft-ii/
[2] Human-level performance in first-person multiplayer games with population-based deep reinforcement learning
    https://arxiv.org/abs/1807.01281
[3] Open-ended Learning in Symmetric Zero-sum Games
    https://arxiv.org/abs/1901.08106
[4] A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning.
[5] Emergent Coordination Through Competition, ICLR2019
[6] AlphaStar ICML2019 workshop slides
[7] AlphaStar Nature Paper
"""
import os
import pickle
import random
import sys
from collections import OrderedDict

import numpy as np

from .base_game_mgr import GameMgr
from .utils import winrates_to_prob
from ..utils import logger
from ..utils.tl_types import is_inherit, LeagueMgrErroMsg
from ..utils.logger import Logger
from ..utils.logger import HumanOutputFormat as HOF


class Individual(object):
  def __init__(self, total_trained_lps):
    # indicates the no. learning periods from root model to current model
    self.total_trained_lps = total_trained_lps
    self.is_historical = False
    self.sampled_as_oppo_count = 0

  def increment_lp(self):
    self.total_trained_lps += 1

  def decide_descendent(self, individuals, players, winrates, cur_player):
    raise NotImplementedError('Do not call base class method')

  def decide_opponent(self, individuals, winrates):
    raise NotImplementedError('Do not call base class method')

  def __str__(self):
    return ('sampled_as_oppo_count: {}, total_trained_lps: {},'
            'is_historical: {}').format(
      self.sampled_as_oppo_count,
      self.total_trained_lps,
      self.is_historical
    )

  @staticmethod
  def find_root(individuals, players, cur_player=None):
    if cur_player is None:
      # This is applicable only when there is only
      # one unique init_model model in the league
      for idx, p in enumerate(players):
        assert isinstance(p, str)
        if p.lower().startswith('none'):
          return individuals[idx]
    else:
      # This is the general procedure that one should
      # recursively find the specific root
      assert cur_player in players, """Unknown current 
      player {}""".format(cur_player)
      tmp_player = cur_player
      while True:
        if tmp_player.lower().startswith('none'):
          return individuals[players.index(tmp_player)]
        split_players = tmp_player.split(':')
        assert len(split_players) == 2, """Illegal player 
        id {}""".format(tmp_player)
        parent_p = split_players[0]
        found = False
        for p in players:
          if p.split(':')[1] == parent_p:
            tmp_player = p
            found = True
            break
        if not found:
          return None
    return None

  @staticmethod
  def find_leaves(individuals, players):
    # TODO(pengsun): O(n^2), could be problematic when len(players) > 2000?
    leaves = []
    for idx, p in enumerate(players):
      assert isinstance(p, str)
      is_leaf = True
      for p2 in players:
        if is_inherit(p, p2):
          is_leaf = False
          break
      if is_leaf:
        leaves.append(individuals[idx])
    return leaves

  @staticmethod
  def find_leaves_with_idx(individuals, players):
    # TODO(pengsun): O(n^2), could be problematic when len(players) > 2000?
    leaves = []
    leaf_idx = []
    for idx, p in enumerate(players):
      assert isinstance(p, str)
      is_leaf = True
      for p2 in players:
        if is_inherit(p, p2):
          is_leaf = False
          break
      if is_leaf:
        leaves.append(individuals[idx])
        leaf_idx.append(idx)
    return leaves, leaf_idx


Exploiter = Individual


class IndivMainAgent(Individual):
  def __init__(self,
               total_trained_lps,
               main_agent_pfsp_prob,
               main_agent_forgotten_prob,
               main_agent_forgotten_me_winrate_thre,
               main_agent_forgotten_ma_winrate_thre,
               **kwargs):
    super(IndivMainAgent, self).__init__(
      total_trained_lps
    )
    self.pfsp_prob = main_agent_pfsp_prob  # e.g., 0.5
    self.forgotten_prob = main_agent_forgotten_prob  # e.g., 0.15
    # e.g., 0.3
    self.forgotten_me_winrate_thre = main_agent_forgotten_me_winrate_thre
    # e.g., 0.7
    self.forgotten_ma_winrate_thre = main_agent_forgotten_ma_winrate_thre

  def decide_descendent(self, individuals, players, winrates,
                        cur_player=None, coin_toss_reset=None):
    # always continue training (but with another model key in LeagueMgr)
    return self

  def decide_opponent(self, individuals, winrates, coin_toss=None):
    coin_toss = coin_toss or random.random()

    if coin_toss < self.pfsp_prob:
      return self._pfsp_opponent(individuals, winrates)

    if coin_toss < self.pfsp_prob + self.forgotten_prob:
      forgotten_oppo = self._forgotten_opponent(individuals, winrates)
      if forgotten_oppo is not None:
        return forgotten_oppo

    return self._sp_opponent(individuals, winrates)

  def _pfsp_opponent(self, individuals, winrates):
    # prioritized fictitious self play
    historical_individuals, historical_winrates = [], []
    for i, w in zip(individuals, winrates):
      if i.is_historical:
        historical_individuals.append(i)
        historical_winrates.append(w)
    if historical_winrates:
      return np.random.choice(historical_individuals,
                              p=winrates_to_prob(historical_winrates,
                                                 weighting="squared"))
    # no historical players, simple use self
    return self

  def _sp_opponent(self, individuals, winrates):
    # self play (just my self)
    return self

  def _forgotten_opponent(self, individuals, winrates):
    # if any forgotten Main Exploiter
    historical_individuals, historical_winrates = [], []
    for i, w in zip(individuals, winrates):
      if (i.is_historical and isinstance(i, Exploiter)):
        if w < self.forgotten_me_winrate_thre:
          historical_individuals.append(i)
          historical_winrates.append(w)

    if historical_winrates:
      return np.random.choice(historical_individuals,
                              p=winrates_to_prob(historical_winrates,
                                                 weighting="squared"))

    def _remove_monotonic_suffix(_winrates, _individuals):
      if not _winrates:
        return _winrates, _individuals
      for i in range(len(_winrates) - 1, 0, -1):
        if _winrates[i - 1] < _winrates[i]:
          return _winrates[:i + 1], _individuals[:i + 1]
      return [], []

    # if any forgotten Main Agent
    historical_individuals, historical_winrates = [], []
    for i, w in zip(individuals, winrates):
      if (i.is_historical and
          isinstance(i, IndivMainAgent)):
        historical_individuals.append(i)
        historical_winrates.append(w)

    tmp_winrates, tmp_individuals = _remove_monotonic_suffix(
      historical_winrates, historical_individuals
    )

    historical_individuals, historical_winrates = [], []
    for i, w in zip(tmp_individuals, tmp_winrates):
      if w < self.forgotten_ma_winrate_thre:
        historical_individuals.append(i)
        historical_winrates.append(w)

    if historical_winrates:
      return np.random.choice(historical_individuals,
                              p=winrates_to_prob(historical_winrates,
                                                 weighting="squared"))

    # finally, not found any forgotten opponent
    return None

  def __str__(self):
    return ('IndivMainAgent, ' + super(IndivMainAgent, self).__str__())


class IndivSKMainAgent(IndivMainAgent):
  def __init__(self,
               total_trained_lps,
               main_agent_pfsp_prob,
               main_agent_forgotten_prob,
               main_agent_forgotten_me_winrate_thre,
               main_agent_forgotten_ma_winrate_thre,
               main_agent_add_to_league_winrate_thre=0.7,
               **kwargs):
    super(IndivSKMainAgent, self).__init__(
      total_trained_lps,
      main_agent_pfsp_prob,
      main_agent_forgotten_prob,
      main_agent_forgotten_me_winrate_thre,
      main_agent_forgotten_ma_winrate_thre,
      **kwargs
    )
    self.add_to_league_winrate_thre = main_agent_add_to_league_winrate_thre

  def decide_descendent(self, individuals, players, winrates,
                        cur_player=None, coin_toss_reset=None):
    # has sufficiently exploited every historical player?
    historical_winrates = [
      w for i, w in zip(individuals, winrates) if i.is_historical
    ]
    print('historical winrates: {}'.format(historical_winrates))
    if historical_winrates:
      w = min(historical_winrates)
      if w > self.add_to_league_winrate_thre:
        # continue training but with another model key in LeagueMgr
        return self
    # continue training with the same model key
    return None


class IndivMainExploiter(Exploiter):
  def __init__(self,
               total_trained_lps,
               main_exploiter_min_lps,
               main_exploiter_max_lps,
               main_exploiter_winrate_thre,
               main_exploiter_reset_winrate_thre,
               main_exploiter_prob_thre,
               **kwargs):
    super(IndivMainExploiter, self).__init__(
      total_trained_lps
    )
    # NOTE: hard-coding. How long you think it should train
    self.min_lps = main_exploiter_min_lps  # e.g., 7
    self.max_lps = main_exploiter_max_lps  # e.g., 9
    self.winrate_thre = main_exploiter_winrate_thre  # e.g., 0.1
    self.reset_winrate_thre = main_exploiter_reset_winrate_thre  # e.g., 0.7
    self.prob_thre = main_exploiter_prob_thre  # e.g., 0.15

  def decide_descendent(self, individuals, players, winrates,
                        cur_player=None, coin_toss_reset=None):
    if self.total_trained_lps < self.min_lps:
      # simply continue training
      return None

    if self.total_trained_lps >= self.max_lps:
      # inherit from the root individual (reset to the initial model)
      return self.find_root(individuals, players, cur_player)

    # has sufficiently exploited the Main Agent?
    active_individuals, active_winrates = zip(*[
      (i, w) for i, w in zip(individuals, winrates)
      if not i.is_historical
         and isinstance(i, IndivMainAgent)
    ])
    assert len(active_individuals) == 1, 'only one active Main Agent'
    w = active_winrates[0]
    if w > self.reset_winrate_thre:
      # inherit from the root individual (reset to the initial model)
      return self.find_root(individuals, players, cur_player)

    # simply continue training
    return None

  def decide_opponent(self, individuals, winrates, coin_toss=None):
    # if the Main Agent is exploitable?
    active_individuals, active_winrates = zip(*[
      (i, w) for i, w in zip(individuals, winrates)
      if not i.is_historical
         and isinstance(i, IndivMainAgent)
    ])
    assert len(active_individuals) == 1, 'only one active Main Agent'
    if active_winrates[0] > self.winrate_thre:
      return active_individuals[0]

    # too strong the active Main Agent, soft picking (either the active one...
    coin_toss = coin_toss or random.random()
    if coin_toss < self.prob_thre:
      return active_individuals[0]
    # ... or a historical one)
    historical_individuals, historical_winrates = zip(*[
      (i, w) for i, w in zip(individuals, winrates)
      if i.is_historical
         and isinstance(i, IndivMainAgent)
    ])
    if historical_winrates:
      return np.random.choice(historical_individuals,
                              p=winrates_to_prob(historical_winrates,
                                                 weighting="variance"))

    # no exploitable historical Main Agent, simply use the current active one
    return active_individuals[0]

  def __str__(self):
    return ('IndivMainExploiter, ' + super(IndivMainExploiter, self).__str__())


class IndivSpecExploiter(IndivMainExploiter):
  def __init__(self,
               total_trained_lps,
               spec_exploiter_min_lps,
               spec_exploiter_max_lps,
               spec_exploiter_winrate_thre,
               spec_exploiter_reset_winrate_thre,
               spec_exploiter_prob_thre,
               **kwargs):
    super(IndivSpecExploiter, self).__init__(
      total_trained_lps,
      spec_exploiter_min_lps,
      spec_exploiter_max_lps,
      spec_exploiter_winrate_thre,
      spec_exploiter_reset_winrate_thre,
      spec_exploiter_prob_thre,
    )


class IndivEvoExploiter(IndivMainExploiter):
  """ Evolutionary Exploiter that the exploiter inherits from a
  strong version of the exploiter with the same pattern, e.g.,
  ml, hl, etc."""
  def __init__(self,
               total_trained_lps,
               evo_exploiter_min_lps,
               evo_exploiter_max_lps,
               evo_exploiter_winrate_thre,
               evo_exploiter_reset_winrate_thre,
               evo_exploiter_prob_thre,
               evo_exploiter_pattern,
               **kwargs):
    super(IndivEvoExploiter, self).__init__(
      total_trained_lps,
      evo_exploiter_min_lps,
      evo_exploiter_max_lps,
      evo_exploiter_winrate_thre,
      evo_exploiter_reset_winrate_thre,
      evo_exploiter_prob_thre,
    )
    self._ee_pattern = evo_exploiter_pattern

  def find_best_by_pattern(self, individuals, players, active_ma_winrates):
    """ for EE only """
    assert len(players) == len(active_ma_winrates), 'assert in find_best_by_pattern fails.'
    best_idx, best_winrate = None, 1.1
    # we are choosing the best EE that can beat MA, so we need active_ma_winrates
    # which are the winrates from the perspective of MA vs others, and then we
    # choose the lowest value if it corresponds to EE
    for idx, player in enumerate(players):
      if hasattr(individuals[idx], '_ee_pattern') and \
        self._ee_pattern == individuals[idx]._ee_pattern and \
        active_ma_winrates[idx] < best_winrate:
        best_idx = idx
        best_winrate = active_ma_winrates[idx]
    if best_idx is not None:
      return individuals[best_idx]
    return None

  def decide_descendent(self, individuals, players, winrates,
                        cur_player=None, coin_toss_reset=None,
                        active_ma_winrates=None):
    """
    :param individuals: all individuals
    :param players: all players
    :param winrates: winrates of the current player vs all players
    :param cur_player: current player
    :param coin_toss_reset:
    :param active_ma_winrates: active_ma_winrates
    :return: descendent
    """
    if self.total_trained_lps < self.min_lps:
      # simply continue training
      return None

    if self.total_trained_lps >= self.max_lps:
      # inherit from the best with the same pattern
      return self.find_best_by_pattern(individuals, players, active_ma_winrates)

    # has sufficiently exploited the Main Agent?
    active_individuals, active_winrates = zip(*[
      (i, w) for i, w in zip(individuals, winrates)
      if not i.is_historical
         and isinstance(i, IndivMainAgent)
    ])
    assert len(active_individuals) == 1, 'only one active Main Agent'
    w = active_winrates[0]
    if w > self.reset_winrate_thre:
      # add self to league and continue training
      return self

    # simply continue training
    return None


class IndivAdaEvoExploiter(IndivMainExploiter):
  """ Adaptive Evolutionary Exploiter that the exploiter
  inherits from a leaf node that is not too weak with the
  same pattern."""
  def __init__(self,
               total_trained_lps,
               ada_evo_exploiter_min_lps,
               ada_evo_exploiter_max_lps,
               ada_evo_exploiter_winrate_thre,
               ada_evo_exploiter_reset_winrate_thre,
               ada_evo_exploiter_prob_thre,
               ada_evo_exploiter_pattern,
               **kwargs):
    super(IndivAdaEvoExploiter, self).__init__(
      total_trained_lps,
      ada_evo_exploiter_min_lps,
      ada_evo_exploiter_max_lps,
      ada_evo_exploiter_winrate_thre,
      ada_evo_exploiter_reset_winrate_thre,
      ada_evo_exploiter_prob_thre,
    )
    self._aee_pattern = ada_evo_exploiter_pattern

  def find_the_right_leaf_by_pattern(self, individuals, players, active_ma_winrates):
    """ for AEE only """
    assert len(players) == len(active_ma_winrates), 'assert in find_best_by_pattern fails.'
    best_idx, best_winrate = None, 1.1
    # we are choosing an appropriate leaf that is comparable with MA but
    # slightly weaker than MA with vs_MA's winrate in [0.2, 0.5]; for the
    # leaves satisfying the condition, we choose the best one, i.e., the
    # mostly approaching 0.5 winrate against MA
    leaves, leaf_idx = self.find_leaves_with_idx(individuals, players)
    for idx in leaf_idx:
      if hasattr(individuals[idx], '_aee_pattern') and \
        self._aee_pattern == individuals[idx]._aee_pattern and \
        0.5 < active_ma_winrates[idx] < 0.8:
        if active_ma_winrates[idx] < best_winrate:
          best_idx = idx
          best_winrate = active_ma_winrates[idx]
    if best_idx is not None:
      return individuals[best_idx]
    return None

  def decide_descendent(self, individuals, players, winrates,
                        cur_player=None, coin_toss_reset=None,
                        active_ma_winrates=None):
    """
    :param individuals: all individuals
    :param players: all players
    :param winrates: winrates of the current player vs all players
    :param cur_player: current player
    :param coin_toss_reset:
    :param active_ma_winrates: active_ma_winrates
    :return: descendent
    """
    if self.total_trained_lps < self.min_lps:
      # simply continue training
      return None

    if self.total_trained_lps >= self.max_lps:
      # inherit from the right leaf with the same pattern
      descendent = self.find_the_right_leaf_by_pattern(
        individuals, players, active_ma_winrates)
      if descendent is None:
        # if no AEE leaves satisfy the condition, reset the AEE
        print(cur_player)
        return self.find_root(individuals, players, cur_player)
      else:
        return descendent

    # has sufficiently exploited the Main Agent?
    active_individuals, active_winrates = zip(*[
      (i, w) for i, w in zip(individuals, winrates)
      if not i.is_historical
         and isinstance(i, IndivMainAgent)
    ])
    assert len(active_individuals) == 1, 'only one active Main Agent'
    w = active_winrates[0]
    if w > self.reset_winrate_thre:
      descendent = self.find_the_right_leaf_by_pattern(
        individuals, players, active_ma_winrates)
      if descendent is None:
        # if no AEE leaves satisfy the condition, reset the AEE
        return self.find_root(individuals, players, cur_player)
      else:
        return descendent

    # simply continue training
    return None


class IndivLeagueExploiter(Exploiter):
  def __init__(self,
               total_trained_lps,
               league_exploiter_min_lps,
               league_exploiter_max_lps,
               league_exploiter_winrate_thre,
               league_exploiter_prob_thre,
               **kwargs):
    super(IndivLeagueExploiter, self).__init__(
      total_trained_lps
    )
    #
    self.min_lps = league_exploiter_min_lps  # e.g., 7
    self.max_lps = league_exploiter_max_lps  # e.g., 9
    self.winrate_thre = league_exploiter_winrate_thre  # e.g., 0.7
    self.prob_thre = league_exploiter_prob_thre  # e..g, 0.25

  def decide_descendent(self, individuals, players, winrates,
                        cur_player=None, coin_toss_reset=None):
    if self.total_trained_lps < self.min_lps:
      # simply continue training
      return None

    # has trained long enough?
    cur_coin_toss_reset = coin_toss_reset or random.random()
    if (self.total_trained_lps >= self.max_lps
        and cur_coin_toss_reset < self.prob_thre):
      # reset to the initial model
      return self.find_root(individuals, players, cur_player)

    # has sufficiently exploited every historical player?
    historical_winrates = [
      w for i, w in zip(individuals, winrates) if i.is_historical
    ]

    # simply continue training
    return None

  def decide_opponent(self, individuals, winrates, coin_toss=None):
    # any historical player in the league to exploit?
    historical_individuals, historical_winrates = [], []
    for i, w in zip(individuals, winrates):
      if i.is_historical:
        historical_individuals.append(i)
        historical_winrates.append(w)
    if historical_winrates:
      return np.random.choice(historical_individuals,
                              p=winrates_to_prob(historical_winrates,
                                                 weighting="linear_capped"))

    # no historical one, simply use the Main Agent
    for i in individuals:
      if (not i.is_historical and
          isinstance(i, IndivMainAgent)):
        return i
    raise ValueError('no active Main Agent!')

  def __str__(self):
    return ('IndivLeagueExploiter, ' +
            super(IndivLeagueExploiter, self).__str__()
    )


class IndivCyclicExploiter(Exploiter):
  def __init__(self,
               total_trained_lps,
               cyclic_exploiter_n_leaves,
               cyclic_exploiter_prob_thre,
               **kwargs):
    super(IndivCyclicExploiter, self).__init__(total_trained_lps)
    # for decide_opponent, e.g., 0.15 or smaller
    self.prob_thre = cyclic_exploiter_prob_thre
    # Should be the same with the no. of zstat files
    self.n_leaves = cyclic_exploiter_n_leaves

  def decide_descendent(self, individuals, players, winrates,
                        cur_player=None, coin_toss_reset=None):
    # cycle to the next model
    return self._find_next_leaf(cur_player, individuals, players)

  def _find_next_leaf(self, current_player, individuals, players):
    leaves = self.find_leaves(individuals, players)
    cyclic_leaves = [
      leaf for leaf in leaves
      if isinstance(leaf, IndivCyclicExploiter)
    ]

    current_indiv = individuals[players.index(current_player)]
    assert current_indiv in cyclic_leaves

    if len(cyclic_leaves) < self.n_leaves:
      return self.find_root(individuals, players)
    historical_cyclic_leaves = [leaf for leaf in cyclic_leaves
                                if leaf.is_historical]
    if historical_cyclic_leaves:
      # the oldest historical leaf
      return historical_cyclic_leaves[0]
    else:
      return current_indiv

  def decide_opponent(self, individuals, winrates, coin_toss=None):
    # if the active Main Agent is exploitable?
    active_individuals, active_winrates = zip(*[
      (i, w) for i, w in zip(individuals, winrates)
      if not i.is_historical
         and isinstance(i, IndivMainAgent)
    ])
    assert len(active_individuals) == 1, 'only one active Main Agent'
    if active_winrates[0] > 0.1:
      return active_individuals[0]

    # too strong the active Main Agent, soft picking (either the active one...
    coin_toss = coin_toss or random.random()
    if coin_toss < self.prob_thre:
      return active_individuals[0]
    # ... or a historical one)
    historical_individuals, historical_winrates = zip(*[
      (i, w) for i, w in zip(individuals, winrates)
      if i.is_historical
         and isinstance(i,IndivMainAgent)
    ])
    if historical_winrates:
      return np.random.choice(historical_individuals,
                              p=winrates_to_prob(historical_winrates,
                                                 weighting="variance"))

    # no exploitable historical Main Agent, simply use the current active one
    return active_individuals[0]

  def __str__(self):
    return ('IndivCyclicExploiter, ' +
            super(IndivCyclicExploiter, self).__str__()
    )


class AEMatchMakingGameMgr(GameMgr):
  """Agent-Exploiter Match Making Game Manager, see [7].

  Implement the Fictitious Self Play (FSP) - Prioritized Fictitious Self Play
   (PFSP) mixed competitive play involving three roles: Main Agent, Main
   Exploiter and League Exploiter. The algorithm is described in [7].
  """
  def __init__(self,
               pgn_file=None,
               log_file='ae.log',
               stat_decay=0.99,
               lrn_id_list=None,
               lrn_role_list=None,
               verbose=0,
               **kwargs):
    super(AEMatchMakingGameMgr, self).__init__(pgn_file,
                                               stat_decay=stat_decay,
                                               verbose=verbose)
    self.logger = (Logger(dir='.', output_formats=[HOF(log_file)]) if log_file
                   else Logger(dir='.', output_formats=[HOF(sys.stdout)]))
    self.logger.set_level(verbose)
    # book-keep the kwargs for an Individual()
    self._indiv_kwargs = kwargs
    # book-keep learner ids
    self._lrn_id_list = lrn_id_list or ['lrngrp0', 'lrngrp1', 'lrngrp2']
    self._lrn_role_list = lrn_role_list or ['MA', 'ME', 'LE']
    assert len(self._lrn_id_list) == len(self._lrn_role_list), """Length of 
        lrn_id_list must be equal to length of lrn_role_list"""
    # population in {player_id: Individual()}, each a Main Agent, a Main
    # Exploiter or LeagueExploiter.
    self._population = OrderedDict()
    # for debugging, not exposed
    self._dbg_coin_toss_oppo = None
    self._dbg_coin_toss_reset = None

  def save(self, checkpoint_dir):
    super(AEMatchMakingGameMgr, self).save(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, 'population')
    with open(filepath, 'wb') as f:
      pickle.dump(self._population, f)

  def load(self, checkpoint_dir):
    super(AEMatchMakingGameMgr, self).load(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, 'population')
    with open(filepath, 'rb') as f:
      self._population = pickle.load(f)

  def add_player(self, p, parent_p=None, learner_id=None, **kwargs):
    if p == parent_p:
      # it means continue training
      self._log_add_player(p, parent_p, learner_id)
      return

    if p in self.players:
      # it means players may use the same init_model
      logger.log('The player {} has already been in the league.'.format(p))
      return

    # do the payoff matrix related updating and inheritance
    super(AEMatchMakingGameMgr, self).add_player(p, parent_p, **kwargs)
    self._inherit_player(p, parent_p)

    if p == 'None:init_model':
      self._add_player_individual(p, parent_p, IndivMainAgent)
      # The initial model named 'None:init_model' is deemed as a historical
      # Main Agent!
      self._population[p].is_historical = True
      self._log_add_player(p, parent_p, learner_id)
      return

    # Add as Main Agent, Main Exploiter and League Exploiter in order
    err_msg = """Error when adding player using learner_id {}. Note: can only 
      add exactly one Main Agent, and other exploiters, including MainExploiter,
      LeagueExploiter, CyclicExploiter.
    """.format(learner_id)
    pattern = None
    if learner_id is not None:
      assert learner_id in self._lrn_id_list, 'unknown learner_id: {}'.format(
        learner_id)
      if self._lrn_role_list[self._lrn_id_list.index(learner_id)] == 'MA':
        role_cls = IndivMainAgent
      elif self._lrn_role_list[self._lrn_id_list.index(learner_id)] == 'ME':
        role_cls = IndivMainExploiter
      elif self._lrn_role_list[self._lrn_id_list.index(learner_id)] == 'SE':
        role_cls = IndivSpecExploiter
      elif self._lrn_role_list[self._lrn_id_list.index(learner_id)] == 'LE':
        role_cls = IndivLeagueExploiter
      elif self._lrn_role_list[self._lrn_id_list.index(learner_id)] == 'CE':
        role_cls = IndivCyclicExploiter
      elif self._lrn_role_list[self._lrn_id_list.index(learner_id)] == 'SK':
        role_cls = IndivSKMainAgent
      elif '-' in self._lrn_role_list[self._lrn_id_list.index(learner_id)]:
        ss = self._lrn_role_list[self._lrn_id_list.index(learner_id)].split('-')
        assert len(ss) == 2, 'role name incorrect.'
        if ss[0] == 'EE':
          role_cls = IndivEvoExploiter
        elif ss[0] == 'AEE':
          role_cls = IndivAdaEvoExploiter
        else:
          raise ValueError(err_msg)
        pattern = ss[-1]
      else:
        raise ValueError(err_msg)
      active_role_count = self._get_current_active_role_count()
      if role_cls == IndivMainAgent:
        assert role_cls not in active_role_count, err_msg
    else:
      role_cls = IndivMainExploiter

    self._add_player_individual(p, parent_p, role_cls, pattern)
    if learner_id is None:
      self._population[p].is_historical = True
    self._log_add_player(p, parent_p, learner_id)
    return

  def _get_current_active_role_count(self):
    active_role_count = {}
    for indiv in self._population.values():
      if not indiv.is_historical:
        if type(indiv) not in active_role_count:
          active_role_count[type(indiv)] = 1
        else:
          active_role_count[type(indiv)] += 1
    return active_role_count

  def _add_player_individual(self, p, parent_p, role_cls, pattern=None):
    # create the individual and add it to population
    assert role_cls in [IndivMainAgent,
                        IndivMainExploiter,
                        IndivLeagueExploiter,
                        IndivCyclicExploiter,
                        IndivSKMainAgent,
                        IndivSpecExploiter,
                        IndivEvoExploiter,
                        IndivAdaEvoExploiter]
    parent_i = None if parent_p is None else self._population[parent_p]

    # inherit from parent individual (if any)
    if role_cls == IndivEvoExploiter:
      individual = role_cls(
        total_trained_lps=0,  # differs from other roles so that EE can reuse min_lps and max_lps
        evo_exploiter_pattern=pattern,
        **self._indiv_kwargs
      )
    elif role_cls == IndivAdaEvoExploiter:
      individual = role_cls(
        total_trained_lps=0,  # differs from other roles so that EE can reuse min_lps and max_lps
        ada_evo_exploiter_pattern=pattern,
        **self._indiv_kwargs
      )
    else:
      individual = role_cls(
        total_trained_lps=0 if parent_i is None else parent_i.total_trained_lps,
        **self._indiv_kwargs
      )
    self._population[p] = individual

    # logging stuff
    self.logger.log(
      '_add_player_individual,',
      'player: {}, individual: {},'.format(p, individual),
      'parent_player: {}, parent_individual: {},'.format(parent_p, parent_i),
      level=logger.DEBUG + 7
    )

  def _inherit_player(self, p, parent_p):
    if parent_p == p or parent_p is None:
      self.logger.log('_inherit_player, skipped,',
                      'p: {},'.format(p),
                      'parent_p: {}'.format(parent_p),
                      level=logger.DEBUG + 7)
      return

    idx = self.players.index(parent_p)
    self.logger.log('_inherit_player,',
                    'idx: {},'.format(idx),
                    'p: {},'.format(p),
                    'parent_p: {}'.format(parent_p),
                    level=logger.DEBUG + 7)

    def _inherit_tail_row_col(mat):
      mat[:, -1] = mat[:, idx]
      mat[-1, :] = mat[idx, :]

    _inherit_tail_row_col(self.finished_match_counter)
    _inherit_tail_row_col(self.all_match_counter)
    _inherit_tail_row_col(self.sum_outcome)

  def _log_add_player(self, p, parent_p, learner_id):
    if p == parent_p:
      # it means continue training
      self.logger.log('add_player, ',
                      'the same player and parent_player: {},'.format(p),
                      'learner_id: {}'.format(learner_id),
                      level=logger.DEBUG + 7)
      return
    self.logger.log('add_player, '
                    'player: {},'.format(p),
                    'parent_player: {}'.format(parent_p),
                    'learner_id: {}'.format(learner_id),
                    level=logger.DEBUG + 7)
    self.logger.log(
      'finished_match_counter:\n{}'.format(self.finished_match_counter),
      level=logger.DEBUG
    )
    self.logger.log('all_match_counter:\n{}'.format(self.all_match_counter),
                    level=logger.DEBUG)
    self.logger.log('sum_outcome:\n{}'.format(self.sum_outcome),
                    level=logger.DEBUG)

  def finish_match(self, rowp, colp, outcome, info, match_id):
    super(AEMatchMakingGameMgr, self).finish_match(rowp, colp, outcome,
                                                   info, match_id)
    self.logger.logp('finish_match,',
                     'rowp: {},'.format(rowp),
                     'colp: {},'.format(colp),
                     'outcome: {},'.format(outcome),
                     'match_id: {}'.format(match_id),
                     level=logger.DEBUG + 2, prob=0.085)

  def get_player(self, current_player):
    indiv = self._population[current_player]
    if indiv.is_historical:
      self.logger.log(f'get_player for historical player {current_player}!',
                      level=logger.WARN)
    indiv.increment_lp()

    # logging before any modification
    self._log_population(level=logger.DEBUG + 7)

    players = list(self._population.keys())
    individuals = list(self._population.values())
    winrates, _, _ = self.get_one_vs_all_info(cur_player=current_player,
                                              other_players=players,
                                              minimal_matches=5)

    if isinstance(indiv, IndivEvoExploiter) or \
       isinstance(indiv, IndivAdaEvoExploiter):
      active_individuals, active_players = zip(*[
        (i, p) for i, p in zip(individuals, players)
        if not i.is_historical and isinstance(i, IndivMainAgent)
      ])
      assert len(active_individuals) == 1, 'only one active Main Agent'
      active_ma_player = active_players[0]
      active_ma_winrates, _, _ = self.get_one_vs_all_info(
        cur_player=active_ma_player, other_players=players, minimal_matches=5)
      new_indiv = indiv.decide_descendent(
        individuals,
        players,
        winrates,
        cur_player=current_player,
        coin_toss_reset=self._dbg_coin_toss_reset,
        active_ma_winrates=active_ma_winrates,
      )
    else:
      new_indiv = indiv.decide_descendent(
        individuals,
        players,
        winrates,
        cur_player=current_player,
        coin_toss_reset=self._dbg_coin_toss_reset,
      )
    if new_indiv is None:
      # indicates "continue training"
      new_player = None
      is_mutate = False
    else:
      new_player = players[individuals.index(new_indiv)]
      is_mutate = True
      # mark the current player as history
      self._population[current_player].is_historical = True

    self.logger.log('get_player,',
                    'cur_player: {},'.format(current_player),
                    'new_player: {},'.format(new_player),
                    'cur_indiv: {},'.format(indiv),
                    'new_indiv: {}'.format(new_indiv),
                    level=logger.DEBUG + 7)
    return new_player, is_mutate

  def get_opponent(self, player, hyperparam):
    indiv = self._population[player]
    if indiv.is_historical:
      self.logger.log(f'get_opponent for historical player {player}!',
                      level=logger.WARN)

    players = list(self._population.keys())
    individuals = list(self._population.values())
    winrates, _, _ = self.get_one_vs_all_info(cur_player=player,
                                              other_players=players,
                                              minimal_matches=5)
    try:
      oppo_indiv = indiv.decide_opponent(individuals, winrates,
                                         self._dbg_coin_toss_oppo)
    except:
      return LeagueMgrErroMsg('decide_opponent fails!')
    oppo_indiv.sampled_as_oppo_count += 1
    oppo_player = players[individuals.index(oppo_indiv)]
    self.logger.logp('get_opponent,'
                     'player: {},'.format(player),
                     'oppo_player: {}'.format(oppo_player),
                     'individual: {}'.format(indiv),
                     'oppo_individual: {}'.format(oppo_indiv),
                     level=logger.DEBUG + 2,
                     prob=0.05)
    return oppo_player

  def get_eval_match(self):
    raise NotImplementedError

  def _log_population(self, level):
    self.logger.log('Report the whole population,',
                    'size: {}'.format(len(self._population)),
                    level=level)
    for p, i in self._population.items():
      self.logger.log('player: {},'.format(p), 'individual: {}'.format(i),
                      level=level)


class NGUMatchMakingGameMgr(AEMatchMakingGameMgr):
  """
  A Never Give Up game mgr that the main agent should not generate new player
  in the league until it can beat all the league players above a winrate thre
  """
  def _add_player_individual(self, p, parent_p, role_cls):
    if role_cls == IndivMainAgent:
      role_cls = IndivSKMainAgent
    super(NGUMatchMakingGameMgr, self)._add_player_individual(
      p, parent_p, role_cls)
