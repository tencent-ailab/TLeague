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
import random
import sys
from enum import IntEnum
from collections import OrderedDict

import numpy as np
from scipy.stats import norm

from .base_game_mgr import GameMgr
from .utils import make_payoff_matrix
from .utils import gnm_solve
from .utils import elo_predict_winrate
from .utils import elo_update_rating
from ..utils import logger
from ..utils.logger import Logger
from ..utils.logger import HumanOutputFormat as HOF
from ..league_mgrs.league_mgr_msg import LeagueMgrErroMsg
from tleague.game_mgr.utils import winrates_to_prob


class PSROGameMgr(GameMgr):
  """ PSRO based game manager. See [3, 4] for what PSRO is. """

  def __init__(self, pgn_file=None, log_file='psro.log', epsilon=0.2,
               self_prob=0.2, verbose=0, **kwargs):
    super(PSROGameMgr, self).__init__(pgn_file, verbose=verbose)
    self.logger = (Logger(dir='.', output_formats=[HOF(log_file)]) if log_file
                   else Logger(dir='.', output_formats=[HOF(sys.stdout)]))
    self.logger.set_level(verbose)
    self.ne = None
    self.epsilon = epsilon
    self.self_prob = self_prob

  def compute_ne(self):
    print('in comput_ne')
    A = self.get_payoff_matrix()
    re, ce = gnm_solve(A)
    self.ne = re
    self.logger.log('NE:' + str(self.players))
    self.logger.log('NE:' + str(A))
    self.logger.log('NE:' + str(self.ne))

  def get_player(self, current_player):
    """ Sampling from current nash equilibrium """
    self.compute_ne()
    idx = int(np.random.choice(len(self.players), 1, p=self.ne))
    self.logger.log('get_player(%s): %s' % (current_player, self.players[idx]),
                    level=logger.DEBUG)
    return self.players[idx], True

  def get_opponent(self, player, dummy_hyperparam):
    """ Sampling from current nash equilibrium with epsilon uniform"""
    if self.ne is None:
      self.compute_ne()
    r = np.random.random()
    if r > self.self_prob + self.epsilon:
      idx = int(np.random.choice(len(self.ne), 1, p=self.ne))
    elif r > self.self_prob:
      idx = int(np.random.choice(len(self.players), 1))
    else:
      idx = len(self.players) - 1
    self.logger.log("get_opponent(%s): %s" % (player, self.players[idx]),
                    level=logger.DEBUG)
    return self.players[idx]

  def get_eval_match(self):
    """ Find least pair of match """
    n_player = len(self.players)
    M = (self.finished_match_counter + self.finished_match_counter.transpose()
         + np.diag([np.inf] * n_player))
    idx = M.argmin()
    i, j = idx // n_player, idx % n_player
    count = self.finished_match_counter[i, j]
    self.logger.log('get_eval_match(): %s, %s with %s matches finished.' % (
      self.players[i], self.players[j], count), level=logger.DEBUG)
    return self.players[i], self.players[j]


class RefCountGameMgr(PSROGameMgr):
  """ get_eval_match counts all the games """

  def get_eval_match(self):
    """ Find least pair of match """
    n_player = len(self.players)
    M = self.all_match_counter + self.all_match_counter.transpose() + np.diag(
      [np.inf] * n_player)
    idx = M.argmin()
    i, j = idx // n_player, idx % n_player
    count = self.finished_match_counter[i, j]
    self.logger.log(
      'get_eval_match(): %s, %s with %s matches finished or ongoing.' % (
        self.players[i], self.players[j], count), level=logger.DEBUG
    )
    return self.players[i], self.players[j]


class PBTEloMatchMakingGameMgr(GameMgr):
  """ PBT Elo Rating based game manager. See [1, 2, 5]

  Allow mutation when accumulating sufficient lp (Learning Period)s """

  class Individual(object):
    """ Individual in a population """

    def __init__(self, eligible_step=1, total_lps=0, elo_rating=0.0):
      self.elo_rating = elo_rating
      self.trained_lps = 0
      self.total_lps = total_lps
      self.eligible_step = eligible_step
      self.is_living = True  # keep track of the living set, see [2]

    def is_eligible(self):
      return self.trained_lps >= self.eligible_step

    def increment_lp(self):
      self.trained_lps += 1
      self.total_lps += 1

    def __str__(self):
      return ("Individual: elo_rating: {}, trained_lps: {}, "
              "total_trained_lps: {}, is_living: {}").format(
        self.elo_rating, self.trained_lps,
        self.total_lps, self.is_living)

  def __init__(self, pgn_file=None, log_file='elo.log', winrate_threshold=0.3,
               verbose=0, eligible_lps=1, **kwargs):
    super(PBTEloMatchMakingGameMgr, self).__init__(pgn_file, verbose=verbose)
    self.logger = (Logger(dir='.', output_formats=[HOF(log_file)]) if log_file
                   else Logger(dir='.', output_formats=[HOF(sys.stdout)]))
    self.logger.set_level(verbose)
    self.winrate_threshold = winrate_threshold
    self.eligible_lps = eligible_lps
    # TODO(pengsun): support warm-start from existing pgn_file?
    self._population = {}  # {player_id: Individual()}

  def add_player(self, p, parent_p=None, **kwargs):
    if p == parent_p:  # means continue training
      return
    super(PBTEloMatchMakingGameMgr, self).add_player(p, parent_p, **kwargs)
    # add new entry for Elo ratings: an living individual that inherits parent
    # elo rating or is simply from-scratch (i.e., being zero)
    individual = PBTEloMatchMakingGameMgr.Individual(self.eligible_lps)
    if parent_p is not None and parent_p in self._population:
      individual.elo_rating = self._population[parent_p].elo_rating
      individual.total_lps = self._population[parent_p].total_lps

    self._population[p] = individual
    self.logger.log('add_player: {},'.format(p),
                    'parent_player:{},'.format(parent_p),
                    'init Elo rating: {},'.format(individual.elo_rating),
                    level=logger.DEBUG + 7)

  def finish_match(self, rowp, colp, outcome, info, match_id):
    super(PBTEloMatchMakingGameMgr, self).finish_match(rowp, colp, outcome,
                                                       info, match_id)
    if outcome is None:  # actor task early abortion, ignore it
      self.logger.logp('finish_match: outcome is None, ignore',
                       level=logger.DEBUG + 2, prob=0.05)
      return
    # don't update the Elo ratings for the same two players
    if rowp == colp:
      self.logger.logp(
        'finish_match: players {}, {}, '.format(rowp, colp),
        'same players, no updating Elo ratings',
        level=logger.DEBUG + 2, prob=0.05
      )
      return
    # update the Elo ratings on-the-fly
    r_row = self._population[rowp].elo_rating
    r_col = self._population[colp].elo_rating
    updated_r_row, updated_r_col = elo_update_rating(r_row, r_col, outcome)
    self._population[rowp].elo_rating = updated_r_row
    self._population[colp].elo_rating = updated_r_col
    self.logger.logp(
      'finish_match: players {}, {}, '.format(rowp, colp),
      'before ratings {}, {}, '.format(r_row, r_col),
      'after ratings: {}, {}'.format(updated_r_row, updated_r_col),
      level=logger.DEBUG + 2, prob=0.05
    )

  def get_player(self, current_player):
    """ Uniformly sample another player (if any) from the population living set,
     and check if current_player can mutate to it. If can, return it and remove
    the current_player from the living set. In any case, returning None always
    means 'continue training'. """
    self.logger.log('cur player: {}'.format(current_player),
                    level=logger.DEBUG + 7)
    # TODO: increment_lp() should be done at on_notify_learner_task_end
    self._population[current_player].increment_lp()
    cur_individual = self._population[current_player]
    if not cur_individual.is_eligible():
      self.logger.log('trained_lps: {} < {}'.format(
        cur_individual.trained_lps, self.eligible_lps), level=logger.DEBUG + 7)
      return None, False

    # choose the candidate
    living_set = {player: individual for player, individual
                  in self._population.items() if individual.is_living}
    living_players = list(living_set.keys())
    assert len(living_players) > 0
    if len(living_players) > 1:  # strictly choose from the others
      living_players.remove(current_player)
      candidate_player = random.choice(living_players)
    else:
      return None, False
    candidate = living_set[candidate_player]
    # decide whether to mutate
    winrate = elo_predict_winrate(cur_individual.elo_rating,
                                  candidate.elo_rating)
    if winrate < self.winrate_threshold:  # cur_player too weak
      new_player = candidate_player
      is_mutate = True
      removed_player = current_player
      self._population[removed_player].is_living = False
      self.logger.log('new_player {},'.format(new_player),
                      'removed_player: {}'.format(removed_player),
                      level=logger.DEBUG + 7)
    else:  # cur_player not bad
      # continue training with cur_player,
      # return None as the "contracted value" with the league mgr
      new_player = None
      is_mutate = False

    self.logger.log(
      'cur player: {}, candidate player: {}, win-rate: {}, '.format(
        current_player, candidate_player, winrate),
      'living set size: {}, '.format(len(living_set)),
      'get_player({}): {}'.format(current_player, new_player),
      level=logger.DEBUG + 7
    )
    self._log_population(level=logger.DEBUG + 7)
    return new_player, is_mutate

  def get_opponent(self, player, hyperparam):
    """ Probabilistic Elo score matching, See [2], Sect 5.4.1. Over
     the WHOLE population, regardless of being dead or alive! """
    winrates = np.array([
      elo_predict_winrate(self._population[player].elo_rating,
                          self._population[player_oppo].elo_rating)
      for player_oppo in self._population.keys()
    ])

    sigma = hyperparam.sigma
    p = norm.pdf(winrates - 0.5, loc=0.0, scale=sigma)
    p = p / p.sum()

    p_oppo = np.random.choice(list(self._population.keys()), p=p)

    self.logger.logp("sigma: {}".format(sigma),
                     "p: {}\n".format(p),
                     "player: {}".format(player),
                     "winrates: {}\n".format(winrates),
                     "get_opponent({}): {}".format(player, p_oppo),
                     level=logger.DEBUG + 2, prob=0.05)
    return p_oppo

  def get_eval_match(self):
    """ Find least pair of match """
    n_player = len(self.players)
    M = self.all_match_counter + self.all_match_counter.transpose() + np.diag(
      [np.inf] * n_player)
    idx = M.argmin()
    i, j = idx // n_player, idx % n_player
    count = self.finished_match_counter[i, j]
    self.logger.logp(
      'get_eval_match(): %s, %s with %s matches finished or ongoing.' % (
        self.players[i], self.players[j], count), level=logger.DEBUG + 2,
      prob=0.05
    )
    return self.players[i], self.players[j]

  def _log_population(self, level):
    self.logger.log('Report the whole population,',
                    'size: {}'.format(len(self._population)),
                    level=level)
    for p, i in self._population.items():
      self.logger.log('player: {}'.format(p), 'indiv: {}'.format(i),
                      level=level)


class PBTPSROGameMgr(GameMgr):
  """ PBT based PSRO.

  Maintains a living set, in which each individual performs PSRO-like numeric
  optimization in parallel. See [1, 2, 3, 4].  """

  class Individual(object):
    """ Individual in a population """

    def __init__(self, eligible_lps):
      self.trained_lps = 0
      self.total_lps = 0
      self.eligible_lps = eligible_lps
      self.is_living = True  # keep track of the living set, see [2]
      self.tasked_opponents = []

    def is_eligible(self):
      return self.trained_lps >= self.eligible_lps

    def increment_lp(self):
      self.trained_lps += 1
      self.total_lps += 1

    def __str__(self):
      return ("Individual: trained_lps: {}, "
              "total_trained_lps: {}, is_living: {}").format(
        self.trained_lps, self.total_lps, self.is_living)

  def __init__(self, pgn_file=None, log_file='pbt_psro.log', verbose=0,
               eligible_lps=2, **kwargs):
    super(PBTPSROGameMgr, self).__init__(pgn_file, verbose=verbose)
    self.logger = (Logger(dir='.', output_formats=[HOF(log_file)]) if log_file
                   else Logger(dir='.', output_formats=[HOF(sys.stdout)]))
    self.logger.set_level(verbose)
    self.eligible_lps = eligible_lps
    # TODO(pengsun): support warm-start from existing pgn_file?
    self._population = {}  # {player_id: Individual()}

  def add_player(self, p, parent_p=None, **kwargs):
    """ Add a new player or continue training from previous player. In both
    cases, recreate the tasked opponents for the input player p. """
    if p != parent_p:
      # really add a new player to the match outcome, count, etc.
      super(PBTPSROGameMgr, self).add_player(p, parent_p, **kwargs)
      # and add the new player to the population
      individual = PBTPSROGameMgr.Individual(self.eligible_lps)
      if parent_p is not None and parent_p in self._population:
        # inherit the total learning period count
        individual.total_lps = self._population[parent_p].total_lps
      self._population[p] = individual

    # To this extent, player p must be available in the living set, we can
    # safely re-create the tasked opponents for p
    self.logger.log('add_player: {},'.format(p),
                    'parent_player: {},'.format(parent_p),
                    level=logger.DEBUG + 7)
    self._make_tasked_opponents(p)

  def get_player(self, current_player):
    """ get the payoff matrix row (living set only) of the current_player and
     check its diversity measure. If big, continue training (return None as
     contracted value); If small, choose a random one from the living set. """

    # update internal book-keeping
    self.logger.log('get_player,', 'cur player: {}'.format(current_player),
                    level=logger.DEBUG + 7)
    # TODO: increment_lp() should be done at on_notify_learner_task_end
    self._population[current_player].increment_lp()

    # check mutation eligibility
    cur_individual = self._population[current_player]
    if not cur_individual.is_eligible():
      self.logger.log('trained_lps: {} < {},'.format(
        cur_individual.trained_lps, self.eligible_lps), 'continue training',
        level=logger.DEBUG + 7)
      return None, False

    # payoff matrix & diversities
    A, living_players = self._get_livingset_payoff_matrix()
    self.logger.log('get_player,', 'living set payoff matrix A: ',
                    level=logger.DEBUG + 7)

    A_relu = A * (A > 0)  # (N, N)
    v_diversity = A_relu.sum(axis=1)  # (N,)
    cur_ind = living_players.index(current_player)
    cur_d = v_diversity[cur_ind]
    max_d = v_diversity.max()
    self.logger.log(str(A), level=logger.DEBUG + 7)
    self.logger.log('get_player,', 'cur_diversity: {},'.format(cur_d),
                    'max_diversity: {},'.format(max_d),
                    'v_diversity: {},'.format(v_diversity),
                    level=logger.DEBUG + 7)

    # check whether it is diverse enough
    # TODO(pengsun): more elaborated method
    RATIO_THRESHOLD = 0.4
    cur_d_big_enough = cur_d > RATIO_THRESHOLD * max_d
    if cur_d_big_enough:
      # diversity measure high enough, continue training.
      new_player = None  # None means to continue training
      is_mutate = False
    else:
      # diversity measure too low, kill it and randomly choose another player
      self._population[current_player].is_living = False
      living_players = self._get_living_players()
      assert len(living_players) > 0
      new_player = random.choice(living_players)
      is_mutate = True
    self.logger.log('get_player,', 'new_player: {},'.format(new_player),
                    level=logger.DEBUG + 7)
    self._log_population(level=logger.DEBUG + 7)
    return new_player, is_mutate

  def get_opponent(self, player, hyperparam):
    """ find the player individual from the living set. Randomly sample from its
    tasked opponents as the opponent. """
    individual = self._population[player]
    assert individual.is_living == True
    assert len(individual.tasked_opponents) > 0
    p_oppo = random.choice(individual.tasked_opponents)
    self.logger.logp(
      "get_opponent,", "oppo: {},".format(p_oppo),
      "tasked_opponents: {},".format(individual.tasked_opponents),
      level=logger.DEBUG + 2, prob=0.005
    )
    return p_oppo

  def _make_tasked_opponents(self, player):
    # extract the payoff matrix for the living players
    A, living_players = self._get_livingset_payoff_matrix()
    self.logger.log('make_tasked_opponents,', 'A: ',
                    level=logger.DEBUG + 7)
    self.logger.log(str(A), level=logger.DEBUG + 7)
    self.logger.log('make_tasked_opponents,',
                    'living_players: {}'.format(living_players),
                    level=logger.DEBUG + 7)

    ind = living_players.index(player)
    vec_payoff = A[ind, :]  # (N,)
    vec_p = np.array(vec_payoff > 0.0, dtype=np.float32)  # (N,)
    self.logger.log('make_tasked_opponents,', 'ind_player: {},'.format(ind),
                    'vec_p: {}'.format(vec_p.tolist()), level=logger.DEBUG + 7)

    # opponents: specific one, or a distribution?
    PROB_SELECT_SPECIFIC = 0.6
    EPS = 0.00001
    if np.all(vec_p < EPS):
      # collect as opponents all the living players except for self
      p_opponents = [p for p in living_players if p != player]
      self.logger.log('make_tasked_opponents,', 'payoff vector all zeros',
                      level=logger.DEBUG + 7)
    else:
      if random.random() < PROB_SELECT_SPECIFIC:
        # sample a single one opponent
        p_opponents = [np.random.choice(living_players, p=vec_p / vec_p.sum())]
        self.logger.log('make_tasked_opponents,', 'a specific opponent',
                        level=logger.DEBUG + 7)
      else:
        # collect multiple opponents with non-zero payoff value
        p_opponents = [player for sign_payoff, player
                       in zip(vec_p, living_players) if sign_payoff > EPS]
        self.logger.log('make_tasked_opponents,', 'multiple opponents',
                        level=logger.DEBUG + 7)

    if not p_opponents:
      p_opponents = living_players
      self.logger.log(
        'make_tasked_opponents,',
        'cannot pick tasked opponents in regular way,',
        'collect all living players', level=logger.DEBUG + 7)

    # re-create tasked opponents: overwrite to it
    self.logger.log('make_tasked_opponents,',
                    'opponents: {}'.format(p_opponents),
                    level=logger.DEBUG + 7)
    self._population[player].tasked_opponents = p_opponents

  def _get_livingset_payoff_matrix(self):
    """ the matrix and the livingset player list are in the same player """
    living_players = self._get_living_players()
    # make the living set payoff sub-matrix
    indices = [self.players.index(p) for p in living_players]
    living_sum_outcome = self.sum_outcome[np.ix_(indices, indices)]
    living_match_count = self.finished_match_counter[np.ix_(indices, indices)]
    living_A = make_payoff_matrix(living_sum_outcome, living_match_count)
    assert living_A.ndim == 2
    assert living_A.shape[0] == living_A.shape[1], 'A shape {}'.format(
      living_A.shape)
    assert living_A.shape[0] == len(living_players), 'len: {}'.format(
      len(living_players))
    return living_A, living_players

  def _get_living_players(self):
    return [player for player, individual in self._population.items()
            if individual.is_living]

  def _log_population(self, level):
    self.logger.log('Report the whole population,',
                    'size: {}'.format(len(self._population)),
                    level=level)
    for p, i in self._population.items():
      self.logger.log('player: {}'.format(p), 'indiv: {}'.format(i),
                      level=level)


class ACMatchMakingGameMgr(GameMgr):
  """ Agent-Competitor match making game manager. See [1, 6].  """

  class LrnObj(IntEnum):
    """ Learning Objective. See [1, 6].
     IMPROVE: improve self by sampling hard competitor and performing RL
     EXPLOIT: exploit other agents, and help them find the weakness """
    IMPROVE = 1
    EXPLOIT = 2

  class Role(IntEnum):
    """ Two possible roles of the player population. See [6].
    AGENT: the learning agent that updates its own NN parameters on and on.
    COMPETITOR: the fixed agent that serves as opponent. """
    AGENT = 1
    COMPETITOR = 2

  class Individual(object):
    """ Individual in a population """

    def __init__(self, role, total_trained_lps):
      self.role = role
      self.total_trained_lps = total_trained_lps
      self.lrn_obj = ACMatchMakingGameMgr.LrnObj.EXPLOIT  # default objective
      self.trained_lps = 0
      self.prev_competitors = {}
      # mean winrates over the previous competitors
      self.prev_avg_competitor_winrate = None

    def increment_lp(self):
      self.trained_lps += 1
      self.total_trained_lps += 1

    def __str__(self):
      _s_role = lambda i: ('AGENT' if i == ACMatchMakingGameMgr.Role.AGENT
                           else 'COMPETITOR')
      _s_lrnobj = lambda i: (
        'IMPROVE' if i == ACMatchMakingGameMgr.LrnObj.IMPROVE else 'EXPLOIT'
      )
      return ("Individual: role: {}, trained_lps: {}, total_trained_lps: {}, "
              "lrn_obj: {}, prev_competitors: {}, "
              "prev_avg_competitor_winrate: {}".format(
        _s_role(self.role),
        self.trained_lps,
        self.total_trained_lps,
        _s_lrnobj(self.lrn_obj),
        list(self.prev_competitors.keys()),
        self.prev_avg_competitor_winrate
      ))

  def __init__(self, pgn_file=None, log_file='ac.log', verbose=0, **kwargs):
    super(ACMatchMakingGameMgr, self).__init__(pgn_file, verbose=verbose)
    self.logger = (Logger(dir='.', output_formats=[HOF(log_file)]) if log_file
                   else Logger(dir='.', output_formats=[HOF(sys.stdout)]))
    self.logger.set_level(verbose)
    self._population = {}  # {player_id: Individual()}, including both the
    # Agents and Competitors

  def add_player(self, p, parent_p=None, **kwargs):
    if p == parent_p:
      # it means continue training
      self.logger.log('add_player:',
                      'the same player and parent_player: {},'.format(p),
                      level=logger.DEBUG + 7)
      return

    # do the payoff matrix related updating and inheritance
    super(ACMatchMakingGameMgr, self).add_player(p, parent_p, **kwargs)
    self._inherit_player(p, parent_p)

    if p == 'None:init_model':
      self._add_to_population_as_competitor(p, parent_p)
    else:
      self._add_to_population_as_agent(p, parent_p)

    self.logger.log(
      'finished_match_counter: {}'.format(self.finished_match_counter),
      level=logger.DEBUG
    )
    self.logger.log('all_match_counter: {}'.format(self.all_match_counter),
                    level=logger.DEBUG)
    self.logger.log('sum_outcome: {}'.format(self.sum_outcome),
                    level=logger.DEBUG)
    self.logger.log('add_player:', 'player: {},'.format(p),
                    'parent_player: {}'.format(parent_p),
                    level=logger.DEBUG + 7)

  def _add_to_population_as_agent(self, p, parent_p):
    """ add p as as AGENT  """

    # add p to Agent set. In our impl, just add p to population and mark it as
    # Agent. Meanwhile, assign it a learning objective (by random sampling)
    individual = ACMatchMakingGameMgr.Individual(
      role=ACMatchMakingGameMgr.Role.AGENT,
      total_trained_lps=0
    )
    # inherit from parent individual...
    if parent_p is not None and parent_p in self._population:
      # ...the trained lps
      individual.total_trained_lps = (
        self._population[parent_p].total_trained_lps
      )
      # ...the competitors
      individual.prev_competitors = self._population[parent_p].prev_competitors
      individual.prev_avg_competitor_winrate = (
        self._population[parent_p].prev_avg_competitor_winrate
      )
    # make learning objective for the individual
    individual.lrn_obj = self._make_lrn_obj(p, parent_p)

    # TODO(pengsun): apply epsilon-greedy to perturb the lrn_obj again?

    parent_i = None if parent_p is None else self._population[parent_p]
    self.logger.log('_add_to_population_as_agent:',
                    'player: {}, indv: {},'.format(p, individual),
                    'parent_player: {}, parent_indv: {},'.format(parent_p,
                                                                 parent_i),
                    level=logger.DEBUG + 7)
    self._population[p] = individual

  def _make_lrn_obj(self, p, parent_p):
    """create a new learning objective or inherit it from the parent"""
    if parent_p == 'None:init_model' or parent_p is None:
      # random sampling
      lrn_obj = (ACMatchMakingGameMgr.LrnObj.IMPROVE if random.random() < 0.65
                 else ACMatchMakingGameMgr.LrnObj.EXPLOIT)
    else:
      lrn_obj = self._population[parent_p].lrn_obj
    return lrn_obj

  def _add_to_population_as_competitor(self, p, parent_p):
    """ add p as COMPETITOR, ignore parent_p  """
    assert p is not None
    individual = ACMatchMakingGameMgr.Individual(
      role=ACMatchMakingGameMgr.Role.COMPETITOR,
      total_trained_lps=0
    )
    parent_i = None if parent_p is None else self._population[parent_p]
    self.logger.log(
      '_add_to_population_as_competitor:',
      'player: {}, individual: {},'.format(p, individual),
      'parent_player: {}, individual: {},'.format(parent_p, parent_i),
      level=logger.DEBUG + 7
    )
    self._population[p] = individual

  def _inherit_player(self, p, parent_p):
    if parent_p == p or parent_p is None:
      self.logger.log('_inherit_player:', 'skipped,', 'p: {},'.format(p),
                      'parent_p: {}'.format(parent_p), level=logger.DEBUG + 7)
      return

    idx = self.players.index(parent_p)
    self.logger.log('_inherit_player:', 'idx: {},'.format(idx),
                    'p: {},'.format(p), 'parent_p: {}'.format(parent_p),
                    level=logger.DEBUG + 7)

    def _inherit_tail_row_col(mat):
      mat[:, -1] = mat[:, idx]
      mat[-1, :] = mat[idx, :]

    _inherit_tail_row_col(self.finished_match_counter)
    _inherit_tail_row_col(self.all_match_counter)
    _inherit_tail_row_col(self.sum_outcome)

  def finish_match(self, rowp, colp, outcome, info, match_id):
    super(ACMatchMakingGameMgr, self).finish_match(rowp, colp, outcome,
                                                   info, match_id)
    self.logger.logp('finish_match:',
                     'rowp: {}'.format(rowp),
                     'colp: {}'.format(colp),
                     'outcome: {}'.format(outcome),
                     'match_id: {}'.format(match_id),
                     level=logger.DEBUG + 2, prob=0.085)

  def get_player(self, current_player):
    """ given current player model key, return a new model key (can be self)
    from which to mutate and continue learning; return None to indicate not
    mutating and continue learning from the current model key."""
    # TODO: increment_lp() should be done at on_notify_learner_task_end
    self._population[current_player].increment_lp()

    # logging before any modification
    self._log_population(level=logger.DEBUG + 7)

    lrn_obj = self._population[current_player].lrn_obj
    if lrn_obj == ACMatchMakingGameMgr.LrnObj.IMPROVE:
      should_mutate = not self._eligible_to_improve(current_player)
    elif lrn_obj == ACMatchMakingGameMgr.LrnObj.EXPLOIT:
      should_mutate = not self._eligible_to_exploit(current_player)
    else:
      raise ValueError('Unknown learning objective {}'.format(lrn_obj))

    if should_mutate:
      agents = self._collect_players_by_role(
        ACMatchMakingGameMgr.Role.AGENT
      )
      candidates = list(agents.keys())
      new_player = random.choice(candidates)
      self.logger.log('get_player:', 'mutate!', level=logger.DEBUG + 7)
    else:
      # return the current player as the new player,
      # let league manager generate a new model key based on it
      new_player = current_player
      self.logger.log('get_player:', 'do NOT mutate!', level=logger.DEBUG + 7)

    # Always add current_player to Competitor set. In our impl, just mark the
    # current player as competitor
    self._population[current_player].role = ACMatchMakingGameMgr.Role.COMPETITOR

    self.logger.log('get_player:', 'cur_player: {},'.format(current_player),
                    'new_player: {}'.format(new_player), level=logger.DEBUG + 7)
    return new_player, should_mutate

  def _eligible_to_improve(self, current_player):
    is_eligible = True
    prev_avg_wr = self._population[current_player].prev_avg_competitor_winrate
    prev_competitors = self._population[current_player].prev_competitors
    new_prev_avg_wr = None
    if prev_competitors:
      # over the previous competitors; ignore the newly added competitors
      new_winrate = self._collect_winrate_vec(current_player,
                                              list(prev_competitors.keys()))
      new_prev_avg_wr = new_winrate.mean()
      if prev_avg_wr is not None and new_prev_avg_wr < prev_avg_wr:
        # too weak the current player, give it up
        is_eligible = False

    # update the agent's competitors when applicable
    competitors = self._collect_players_by_role(
      ACMatchMakingGameMgr.Role.COMPETITOR
    )
    winrate, avg_wr = None, None
    if competitors:
      self._population[current_player].prev_competitors = competitors
      # update the metric - the prev_avg_competitor_winrate
      winrate = self._collect_winrate_vec(current_player,
                                          list(competitors.keys()))
      avg_wr = winrate.mean()
      self._population[current_player].prev_avg_competitor_winrate = avg_wr

    self.logger.log(
      '_eligible_to_improve:',
      'cur_player: {},'.format(current_player),
      'prev_competitors: {}'.format(list(prev_competitors.keys())),
      'prev_avg_wr: {},'.format(prev_avg_wr),
      'new_prev_avg_wr: {}'.format(new_prev_avg_wr),
      'competitors: {},'.format(list(competitors.keys())),
      'avg_wr: {},'.format(avg_wr),
      'winrate: {},'.format(winrate),
      'is_eligible: {}'.format(is_eligible),
      level=logger.DEBUG + 7
    )
    return is_eligible

  def _eligible_to_exploit(self, current_player):
    agents = self._collect_players_by_role(
      ACMatchMakingGameMgr.Role.AGENT
    )
    is_eligible = True
    winrate, max_winrate = None, None
    if agents:
      winrate = self._collect_winrate_vec(current_player,
                                          list(agents.keys()))
      max_winrate = winrate.max()
      if max_winrate < 0.485:
        # too weak the current player, give it up
        is_eligible = False

    trained_lps = self._population[current_player].trained_lps
    if trained_lps >= 7:
      # trained too long, give it up
      is_eligible = False

    self.logger.log('_eligible_to_exploit:',
                    'cur_player: {},'.format(current_player),
                    'agents: {},'.format(list(agents.keys())),
                    'winrate: {},'.format(winrate),
                    'max_winrate: {},'.format(max_winrate),
                    'trained_lps: {}'.format(trained_lps),
                    'is_eligible: {}'.format(is_eligible),
                    level=logger.DEBUG + 7)
    return is_eligible

  def get_opponent(self, player, hyperparam):
    """ get the opponent for current player """
    lrn_obj = self._population[player].lrn_obj
    # _s_lrnobj = lambda i: (
    #  'IMPROVE' if i == ACMatchMakingGameMgr.LrnObj.IMPROVE else 'EXPLOIT'
    # )
    # self.logger.logp('get_opponent:', 'lrn_obj: {}'.format(_s_lrnobj(lrn_obj)),
    #                 level=logger.DEBUG + 2, prob=0.085)

    if lrn_obj == ACMatchMakingGameMgr.LrnObj.IMPROVE:
      return self._get_opponent_for_improve(player, hyperparam)
    elif lrn_obj == ACMatchMakingGameMgr.LrnObj.EXPLOIT:
      return self._get_opponent_for_exploit(player, hyperparam)
    else:
      raise ValueError('Unknown learning objective {}'.format(lrn_obj))

  def _get_opponent_for_improve(self, player, hyperparam):
    del hyperparam

    # collect all competitors
    competitors = self._collect_players_by_role(
      ACMatchMakingGameMgr.Role.COMPETITOR
    )
    if not competitors:
      self.logger.log(
        'get_opponent_for_improve:',
        'empty competitors, return self {}'.format(player),
        level=logger.DEBUG + 2
      )
      return player

    # make the matching probabilities over these competitors
    winrate = self._collect_winrate_vec(player, list(competitors.keys()))
    c_winrate = 1 - winrate  # NOTE: it's the win-rates of the competitors!
    nl_trans = lambda x: x * x  # [0, 1] \mapsto [0, 1], to amplify the winrate
    prob = nl_trans(c_winrate)
    prob = prob / prob.sum()

    # sample the opponent player
    candidates = list(competitors.keys())
    if isinstance(prob, np.ndarray) and prob.shape[0] > 1:
      p_oppo = np.random.choice(candidates, p=prob)
    else:
      p_oppo = candidates[0]
    self.logger.logp(
      'get_opponent_for_improve:',
      'player: {},'.format(player),
      'oppo_player: {},'.format(p_oppo),
      'c_winrate: {},'.format(c_winrate),
      'prob: {}'.format(prob),
      level=logger.DEBUG + 2,
      prob=0.05
    )
    return p_oppo

  def _get_opponent_for_exploit(self, player, hyperparam):
    del hyperparam

    # collect all agents
    agents = self._collect_players_by_role(ACMatchMakingGameMgr.Role.AGENT)
    assert agents, 'empty Learning Agents, something wrong'

    # sample the opponent player uniformly at random
    p_oppo = random.choice(list(agents.keys()))
    self.logger.logp(
      'get_opponent_for_exploit:',
      'player: {},'.format(player),
      'oppo_player: {}'.format(p_oppo),
      level=logger.DEBUG + 2,
      prob=0.05
    )
    return p_oppo

  def get_eval_match(self):
    raise NotImplementedError

  def _collect_players_by_role(self, role):
    return {
      pp: ii for pp, ii in self._population.items() if ii.role == role
    }

  def _collect_winrate_vec(self, cur_player, other_players, minimal_matches=10):
    winrate, sum_outcome, match_count = self.get_one_vs_all_info(
      cur_player,
      other_players,
      minimal_matches
    )
    self.logger.log('winrate: {},'.format(winrate),
                    'sum_outcome: {},'.format(sum_outcome),
                    'match_count: {}'.format(match_count),
                    level=logger.DEBUG)
    return winrate

  def _log_population(self, level):
    self.logger.log('Report the whole population,',
                    'size: {}'.format(len(self._population)),
                    level=level)
    for p, i in self._population.items():
      self.logger.log('player: {},'.format(p), 'individual: {}'.format(i),
                      level=level)


class ACFixedObjRatioMatchMakingGameMgr(ACMatchMakingGameMgr):
  """ Agent-Competitor match making game manager with Fixed Learning Objectives
  Ratio. See [1, 6].  """
  def __init__(self, pgn_file=None, log_file='acfor.log', verbose=0, **kwargs):
    super(ACFixedObjRatioMatchMakingGameMgr, self).__init__(
      pgn_file, log_file, verbose)

  def _make_lrn_obj(self, p, parent_p):
    """Make a fixed IMPROVE/EXPLOIT ratio. 3 IMPROVE, 2 EXPLOIT """
    lrn_obj_count = self._compute_lrn_obj_count()
    n_improve = lrn_obj_count[ACMatchMakingGameMgr.LrnObj.IMPROVE]
    n_exploit = lrn_obj_count[ACMatchMakingGameMgr.LrnObj.EXPLOIT]
    eps = 0.000001
    i_to_e = float(n_improve) / float(n_exploit + eps)
    if i_to_e > (3.0/2.0 - eps):
      # too many IMPROVE, add EXPLOIT
      lrn_obj = ACMatchMakingGameMgr.LrnObj.EXPLOIT
    else:
      # insufficient IMPROVE, add IMPROVE
      lrn_obj = ACMatchMakingGameMgr.LrnObj.IMPROVE
    self.logger.log(
      '_make_lrn_obj:',
      'n_improve: {}, n_exploit: {}'.format(n_improve, n_exploit),
      level=logger.DEBUG + 7)
    return lrn_obj

  def _compute_lrn_obj_count(self):
    loc = {
      ACMatchMakingGameMgr.LrnObj.EXPLOIT: 0,
      ACMatchMakingGameMgr.LrnObj.IMPROVE: 0
    }
    for _, indv in self._population.items():
      if indv.role == ACMatchMakingGameMgr.Role.AGENT:
        loc[indv.lrn_obj] += 1
    return loc


class PBTACMatchMakingGameMgr(GameMgr):
  """ Agent-Competitor match making game manager. See [1, 6].  """

  class LrnObj(IntEnum):
    """ Learning Objective. See [1, 6].
     IMPROVE: improve self by sampling hard competitor and performing RL
     EXPLOIT: exploit other agents, and help them find the weakness """
    IMPROVE = 1
    EXPLOIT = 2

  class Role(IntEnum):
    """ Two possible roles of the player population. See [6].
    AGENT: the learning agent that updates its own NN parameters on and on.
    COMPETITOR: the fixed agent that serves as opponent. """
    AGENT = 1
    COMPETITOR = 2

  class Individual(object):
    """ Individual in a population """

    def __init__(self, role, total_trained_lps, lrn_obj):
      self.role = role
      self.total_trained_lps = total_trained_lps
      self.lrn_obj = lrn_obj
      self.trained_lps = 0
      # self.eligible_lps = eligible_lps
      self.prev_competitors = {}
      # mean winrates over the previous competitors
      self.prev_avg_competitor_winrate = None

    def increment_lp(self):
      self.trained_lps += 1
      self.total_trained_lps += 1

    def __str__(self):
      _s_role = lambda i: ('AGENT' if i == ACMatchMakingGameMgr.Role.AGENT
                           else 'COMPETITOR')
      _s_lrnobj = lambda i: (
        'IMPROVE' if i == ACMatchMakingGameMgr.LrnObj.IMPROVE else 'EXPLOIT'
      )
      return ("Individual: role: {}, trained_lps: {}, total_trained_lps: {}, "
              "lrn_obj: {}, prev_competitors: {}, "
              "prev_avg_competitor_winrate: {}".format(
        _s_role(self.role),
        self.trained_lps,
        self.total_trained_lps,
        _s_lrnobj(self.lrn_obj),
        list(self.prev_competitors.keys()),
        self.prev_avg_competitor_winrate
      ))

  def __init__(self, pgn_file=None, log_file='ac.log', verbose=0,
               eligible_lps_by_lobj=None, **kwargs):
    super(PBTACMatchMakingGameMgr, self).__init__(pgn_file, verbose=verbose)
    self.logger = (Logger(dir='.', output_formats=[HOF(log_file)]) if log_file
                   else Logger(dir='.', output_formats=[HOF(sys.stdout)]))
    self.logger.set_level(verbose)
    self.eligible_lps_by_lobj = eligible_lps_by_lobj or [-1, -1]
    self._population = {}  # {player_id: Individual()}, including both the
    # Agents and Competitors

  def add_player(self, p, parent_p=None, **kwargs):
    if p == parent_p:
      # it means continue training
      self.logger.log('add_player:',
                      'the same player and parent_player: {},'.format(p),
                      level=logger.DEBUG + 7)
      return

    # do the payoff matrix related updating and inheritance
    super(PBTACMatchMakingGameMgr, self).add_player(p, parent_p, **kwargs)
    self._inherit_player(p, parent_p)

    if p == 'None:init_model':
      self._add_to_population_as_competitor(p, parent_p)
    else:
      self._add_to_population_regular(p, parent_p)

    self.logger.log('add_player:', 'player: {},'.format(p),
                    'parent_player: {}'.format(parent_p),
                    level=logger.DEBUG + 7)

  def _add_to_population_regular(self, p, parent_p):
    """ add p as as AGENT, add parent_p as COMPETITOR  """
    # add parent_p to Competitor set when applicable. In our impl, just mark
    # the player as competitor
    if parent_p is not None:
      assert parent_p in self._population, "{} not in population {}".format(
        parent_p, self._population)
      self._population[parent_p].role = PBTACMatchMakingGameMgr.Role.COMPETITOR

    # add p to Agent set. In our impl, just add p to population and mark it as
    # Agent. Meanwhile, assign it a learning objective (by random sampling)
    lrn_obj = (PBTACMatchMakingGameMgr.LrnObj.IMPROVE if random.random() < 0.65
               else PBTACMatchMakingGameMgr.LrnObj.EXPLOIT)

    individual = PBTACMatchMakingGameMgr.Individual(
      role=PBTACMatchMakingGameMgr.Role.AGENT,
      total_trained_lps=0,
      lrn_obj=lrn_obj
    )
    # inherit the parent individual
    if parent_p is not None and parent_p in self._population:
      individual.total_trained_lps = (
        self._population[parent_p].total_trained_lps
      )
      individual.prev_competitors = self._population[parent_p].prev_competitors
      individual.prev_avg_competitor_winrate = (
        self._population[parent_p].prev_avg_competitor_winrate
      )

    self.logger.log('_add_to_population_as_agent:',
                    'individual: {},'.format(individual),
                    level=logger.DEBUG + 7)
    self._population[p] = individual

  def _add_to_population_as_competitor(self, p, parent_p):
    """ add p as COMPETITOR, ignore parent_p  """
    del parent_p
    assert p is not None
    individual = PBTACMatchMakingGameMgr.Individual(
      role=PBTACMatchMakingGameMgr.Role.COMPETITOR,
      total_trained_lps=0,
      lrn_obj=PBTACMatchMakingGameMgr.LrnObj.EXPLOIT  # arbitrary
    )
    self.logger.log('_add_to_population_as_competitor:',
                    'individual: {},'.format(individual),
                    level=logger.DEBUG + 7)
    self._population[p] = individual

  def _inherit_player(self, p, parent_p):
    if parent_p == p or parent_p is None:
      return
    idx = self.players.index(p)

    def _inherit_tail_row_col(mat):
      mat[:, -1] = mat[:, idx]
      mat[-1, :] = mat[idx, :]

    _inherit_tail_row_col(self.finished_match_counter)
    _inherit_tail_row_col(self.all_match_counter)
    _inherit_tail_row_col(self.sum_outcome)

  def finish_match(self, rowp, colp, outcome, info, match_id):
    super(PBTACMatchMakingGameMgr, self).finish_match(rowp, colp, outcome,
                                                      info, match_id)
    self.logger.logp('finish_match:',
                     'rowp: {}'.format(rowp),
                     'colp: {}'.format(colp),
                     'outcome: {}'.format(outcome),
                     'match_id: {}'.format(match_id),
                     level=logger.DEBUG + 2, prob=0.085)

  def get_player(self, current_player):
    """ given current player model key, return a new model key (can be self)
    from which to mutate and continue learning; return None to indicate not
    mutating and continue learning from current model key."""
    # TODO: increment_lp() should be done at on_notify_learner_task_end
    self._population[current_player].increment_lp()

    lrn_obj = self._population[current_player].lrn_obj
    if lrn_obj == PBTACMatchMakingGameMgr.LrnObj.IMPROVE:
      should_mutate = not self._eligible_to_improve(current_player)
    elif lrn_obj == PBTACMatchMakingGameMgr.LrnObj.EXPLOIT:
      should_mutate = not self._eligible_to_exploit(current_player)
    else:
      raise ValueError('Unknown learning objective {}'.format(lrn_obj))

    if should_mutate:
      agents = self._collect_players_by_role(
        PBTACMatchMakingGameMgr.Role.AGENT
      )
      candidates = list(agents.keys())
      new_player = random.choice(candidates)
      self.logger.log('get_player:', 'mutate!', level=logger.DEBUG + 7)
    else:
      # return the current player as the new player,
      # let league manager generate a new model key based on it
      new_player = current_player
      self.logger.log('get_player:', 'do NOT mutate!', level=logger.DEBUG + 7)
    self.logger.log('get_player:', 'cur_player: {},'.format(current_player),
                    'new_player: {}'.format(new_player), level=logger.DEBUG + 7)
    self._log_population(level=logger.DEBUG + 7)
    return new_player, should_mutate

  def _eligible_to_improve(self, current_player):
    is_eligible = True
    if self._population[current_player].trained_lps < self.eligible_lps_by_lobj[
      PBTACMatchMakingGameMgr.LrnObj.IMPROVE - 1]:
      is_eligible = False
      return is_eligible

    prev_avg_wr = self._population[current_player].prev_avg_competitor_winrate
    prev_competitors = self._population[current_player].prev_competitors
    new_prev_avg_wr = None
    if prev_competitors:
      # over the previous competitors; ignore the newly added competitors
      new_winrate = self._collect_winrate_vec(current_player,
                                              list(prev_competitors.keys()))
      new_prev_avg_wr = new_winrate.mean()
      if prev_avg_wr is not None and new_prev_avg_wr < prev_avg_wr:
        # too weak the current player, give it up
        is_eligible = False

    # update the competitors
    competitors = self._collect_players_by_role(
      PBTACMatchMakingGameMgr.Role.COMPETITOR
    )
    assert competitors, "empty competitors {}".format(competitors)
    self._population[current_player].prev_competitors = competitors
    # update the metric - the prev_avg_competitor_winrate
    winrate = self._collect_winrate_vec(current_player,
                                        list(competitors.keys()))
    avg_wr = winrate.mean()
    self._population[current_player].prev_avg_competitor_winrate = avg_wr

    self.logger.log(
      '_eligible_to_improve:',
      'cur_player: {},'.format(current_player),
      'prev_competitors: {}'.format(list(prev_competitors.keys())),
      'prev_avg_wr: {},'.format(prev_avg_wr),
      'new_prev_avg_wr: {}'.format(new_prev_avg_wr),
      'competitors: {},'.format(list(competitors.keys())),
      'avg_wr: {},'.format(avg_wr),
      'winrate: {},'.format(winrate),
      'is_eligible: {}'.format(is_eligible),
      level=logger.DEBUG + 7
    )
    return is_eligible

  def _eligible_to_exploit(self, current_player):
    agents = self._collect_players_by_role(
      PBTACMatchMakingGameMgr.Role.AGENT
    )
    is_eligible = True
    if self._population[current_player].trained_lps < self.eligible_lps_by_lobj[
      PBTACMatchMakingGameMgr.LrnObj.EXPLOIT - 1]:
      is_eligible = False
      return is_eligible

    winrate, max_winrate = None, None
    if agents:
      winrate = self._collect_winrate_vec(current_player,
                                          list(agents.keys()))
      max_winrate = winrate.max()
      if max_winrate < 0.485:
        # too weak the current player, give it up
        is_eligible = False

    trained_lps = self._population[current_player].trained_lps
    if trained_lps >= 7:
      # trained too long, give it up
      is_eligible = False

    self.logger.log('_eligible_to_exploit:',
                    'cur_player: {},'.format(current_player),
                    'agents: {},'.format(list(agents.keys())),
                    'winrate: {},'.format(winrate),
                    'max_winrate: {},'.format(max_winrate),
                    'trained_lps: {}'.format(trained_lps),
                    'is_eligible: {}'.format(is_eligible),
                    level=logger.DEBUG + 7)
    return is_eligible

  def get_opponent(self, player, hyperparam):
    """ get the opponent for current player """
    lrn_obj = self._population[player].lrn_obj
    # _s_lrnobj = lambda i: (
    #  'IMPROVE' if i == ACMatchMakingGameMgr.LrnObj.IMPROVE else 'EXPLOIT'
    # )
    # self.logger.logp('get_opponent:', 'lrn_obj: {}'.format(_s_lrnobj(lrn_obj)),
    #                 level=logger.DEBUG + 2, prob=0.085)

    if lrn_obj == PBTACMatchMakingGameMgr.LrnObj.IMPROVE:
      return self._get_opponent_for_improve(player, hyperparam)
    elif lrn_obj == PBTACMatchMakingGameMgr.LrnObj.EXPLOIT:
      return self._get_opponent_for_exploit(player, hyperparam)
    else:
      raise ValueError('Unknown learning objective {}'.format(lrn_obj))

  def _get_opponent_for_improve(self, player, hyperparam):
    del hyperparam

    # collect all competitors
    competitors = self._collect_players_by_role(
      PBTACMatchMakingGameMgr.Role.COMPETITOR
    )
    if not competitors:
      self.logger.log(
        'get_opponent_for_improve:',
        'empty competitors, return self {}'.format(player),
        level=logger.DEBUG + 2
      )
      return player

    # make the matching probabilities over these competitors
    winrate = self._collect_winrate_vec(player, list(competitors.keys()))
    c_winrate = 1 - winrate  # NOTE: it's the win-rates of the competitors!
    nl_trans = lambda x: x * x  # [0, 1] \mapsto [0, 1], to emptily the winrate
    prob = nl_trans(c_winrate)
    prob = prob / prob.sum()

    # sample the opponent player
    candidates = list(competitors.keys())
    if isinstance(prob, np.ndarray) and prob.shape[0] > 1:
      p_oppo = np.random.choice(candidates, p=prob)
    else:
      p_oppo = candidates[0]
    self.logger.logp(
      'get_opponent_for_improve:',
      'player: {},'.format(player),
      'oppo_player: {},'.format(p_oppo),
      'c_winrate: {},'.format(c_winrate),
      'prob: {}'.format(prob),
      level=logger.DEBUG + 2,
      prob=0.05
    )
    return p_oppo

  def _get_opponent_for_exploit(self, player, hyperparam):
    del hyperparam

    # collect all agents
    agents = self._collect_players_by_role(PBTACMatchMakingGameMgr.Role.AGENT)
    assert agents, 'empty Learning Agents, something wrong'

    # sample the opponent player uniformly at random
    p_oppo = random.choice(list(agents.keys()))
    self.logger.logp(
      'get_opponent_for_exploit:',
      'player: {},'.format(player),
      'oppo_player: {}'.format(p_oppo),
      level=logger.DEBUG + 2,
      prob=0.05
    )
    return p_oppo

  def get_eval_match(self):
    return NotImplementedError

  def _collect_players_by_role(self, role):
    return {
      pp: ii for pp, ii in self._population.items() if ii.role == role
    }

  def _collect_winrate_vec(self, cur_player, other_players):
    # row vector
    ind_row = [self.players.index(cur_player)]
    ind_cols = [self.players.index(p) for p in other_players]
    r_sum_outcome = self.sum_outcome[np.ix_(ind_row, ind_cols)].squeeze()
    r_match_count = (
      self.finished_match_counter[np.ix_(ind_row, ind_cols)].squeeze()
    )
    # column vector
    ind_rows = [self.players.index(p) for p in other_players]
    ind_col = [self.players.index(cur_player)]
    c_sum_outcome = self.sum_outcome[np.ix_(ind_rows, ind_col)].squeeze()
    c_match_count = (
      self.finished_match_counter[np.ix_(ind_rows, ind_col)].squeeze()
    )
    # enforce symmetry by taking average
    sum_outcome = (r_sum_outcome + c_sum_outcome) / 2.0
    match_count = (r_match_count + c_match_count) / 2.0
    # outcome [-1, +1} -> winrate [0, 1]
    winrate = (1.0 + sum_outcome / (match_count + 0.000001)) / 2.0
    return winrate

  def _log_population(self, level):
    self.logger.log('Report the whole population,',
                    'size: {}'.format(len(self._population)),
                    level=level)
    for p, i in self._population.items():
      self.logger.log('player: {},'.format(p), 'individual: {}'.format(i),
                      level=level)


class RandomGameMgr(GameMgr):
  """ Example game manager. """

  def get_player(self, current_player):
    """ Random Sampling from current players """
    return random.choice(self.players), True

  def get_opponent(self, player, dummy_hyperparam):
    """ Random Sampling from current players"""
    return random.choice(self.players)

  def get_eval_match(self):
    """ Random Sampling from current players """
    return random.sample(self.players, 2)


class SelfPlayGameMgr(GameMgr):
  """Simple SelfPlay game manager.

  Randomly sample a historical player as the opponent(s).
  Almost the Fictitious SelfPlay algorithm."""

  def __init__(self, max_n_players=100, **kwargs):
    super(SelfPlayGameMgr, self).__init__(**kwargs)
    self._max_n_players = max_n_players

  def get_player(self, current_player):
    # Return the latest player
    return self.players[-1], False

  def get_opponent(self, player, dummy_hyperparam):
    # Random sampling from the latest N players
    N = self._max_n_players  # shorter name
    n_player = len(self.players)
    if n_player >= N:
      return random.choice(self.players[-N:])
    else:
      return random.choice(self.players[0:max(1, n_player - 1)])

  def get_eval_match(self):
    # Random Sampling from current players
    return self.players[-1], random.choice(self.players)


class PFSPGameMgr(GameMgr):
  """ Prioritized Fiticious Self-Play for symmetric games.
  """

  def __init__(self, sp=True, **kwargs):
    super(PFSPGameMgr, self).__init__(**kwargs)
    self._sp = sp

  def get_player(self, current_player):
    # Return the latest player
    return self.players[-1], False

  def _pfsp_opponent(self, opponents, winrates):
    # prioritized fictitious self play
    selected_oppo = np.random.choice(
      opponents,
      p=winrates_to_prob(winrates, weighting="squared"))
    return selected_oppo

  def get_opponent(self, player, hyperparam):
    if self._sp:
      opponents = self.players
    else:
      if len(self.players) > 1:
        opponents = self.players[:-1]
      else:
        opponents = self.players
    winrate, _, _ = self.get_one_vs_all_info(
      cur_player=player,
      other_players=opponents,
      minimal_matches=5)
    return self._pfsp_opponent(opponents, winrate)


class FixedInitOppoGameMgr(GameMgr):
  """ A game manager fixing the oppoent as the init_model. """

  def get_player(self, current_player):
    """ Return the latest player """
    return self.players[-1], False

  def get_opponent(self, player, dummy_hyperparam):
    """ Return the init model """
    return self.players[0]

  def get_eval_match(self):
    """ Random Sampling from current players """
    return self.players[-1], random.choice(self.players)
