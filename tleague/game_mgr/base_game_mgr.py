#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 20190311 by qing
# 20190404 modified by jc
# 20190417 modified by pengsun (we don't actually need this, leave it to git)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
from time import time
from copy import deepcopy

import os
import pickle
import numpy as np

from .utils import make_payoff_matrix
from tleague.utils import logger


class GameMgr(object):
  """ Base Class for a full Game Manager """

  def __init__(self, pgn_file=None, stat_decay=None, verbose=0, **kwargs):
    self._stat_decay = stat_decay
    self._verbose = verbose
    self._use_player2order = False #True

    self.players = []
    if self._use_player2order:
      # TODO(qing): Is this equivalent to self.players.index(p)?
      self.player2order = {}
    # (row_player, col_player): [outcomes], where outcome in {-1, 0, 1}
    self.finished_matches = {}
    # (row_player, col_player): {match_id: start_time}
    self.ongoing_matches = {}
    self.finished_match_counter = None  # np.array
    self.all_match_counter = None  # np.array
    self.sum_outcome = None  # np.array
    self.avg_match_time = None
    self.pgn_file = open(pgn_file, 'w') if pgn_file else None

  def save(self, checkpoint_dir):
    filepath = os.path.join(checkpoint_dir, 'league_mgr.payoff')
    with open(filepath, 'wb') as f:
      pickle.dump(self.players, f)
      pickle.dump(self.finished_matches, f)
      pickle.dump(self.finished_match_counter, f)
      pickle.dump(self.sum_outcome, f)

  def load(self, checkpoint_dir):
    filepath = os.path.join(checkpoint_dir, 'league_mgr.payoff')
    with open(filepath, 'rb') as f:
      players = pickle.load(f)
      finished_matches = pickle.load(f)
      finished_match_counter = pickle.load(f)
      sum_outcome = pickle.load(f)

    model_keys = set()
    with open(os.path.join(checkpoint_dir, 'filename.list'), 'rt') as f:
      for model_fn in f:
        if model_fn.strip().endswith('.model'):
          # an eligible model_fn line looks like xxxx:yyyy_timestamp.model
          model_keys.add(model_fn.strip().split('.')[0][:-15])

    missing_idx = [i for i, p in enumerate(players) if p not in model_keys]
    if len(missing_idx) > 0:
      print('Players {} is missing in model pool'.format
            ([players[i] for i in missing_idx]))
    finished_match_counter = np.delete(finished_match_counter, missing_idx, 0)
    finished_match_counter = np.delete(finished_match_counter, missing_idx, 1)
    sum_outcome = np.delete(sum_outcome, missing_idx, 0)
    sum_outcome = np.delete(sum_outcome, missing_idx, 1)
    self.players = [p for p in players if p in model_keys]
    if self._use_player2order:
      self.player2order = dict \
        ([(p, i) for i, p in enumerate(self.players)])
    self.finished_matches = {(rp, cp): c for (rp, cp), c in finished_matches.items()
                                 if rp in model_keys and cp in model_keys}
    for p1 in self.players:
      for p2 in self.players:
        self.ongoing_matches[(p1, p2)] = {}
        self.ongoing_matches[(p2, p1)] = {}
    self.finished_match_counter = finished_match_counter
    self.sum_outcome = sum_outcome
    self.all_match_counter = deepcopy(finished_match_counter)

  def pair2idx(self, rowp, colp):
    assert rowp in self.players
    assert colp in self.players
    if self._use_player2order:
      return self.player2order[rowp], self.player2order[colp]
    else:
      return self.players.index(rowp), self.players.index(colp)

  def _check_player2order(self):
    """ Check self.player2order """
    if self._use_player2order:
      for i in range(len(self.players)):
        p = self.players[i]
        assert p in self.player2order.keys()
        assert self.player2order[p] == i

  def add_player(self, p, parent_p=None, **kwargs):
    """ Add a new player `p` to the player pool with parent `parent_p`.

    Args:
      p: player (model key)
      parent_p: parent player (model key)
    """
    if p == parent_p:
      return

    assert p not in self.players, 'p={}'.format(p)
    n_players = len(self.players)
    self.players.append(p)
    if self._use_player2order:
      self.player2order[p] = n_players
    if n_players == 0:
      self.finished_match_counter = np.zeros((1, 1), np.float)  # can be decayed
      self.all_match_counter = np.zeros((1, 1), np.int)
      self.sum_outcome = np.zeros((1, 1), np.float)  # can be decayed
    else:
      self.finished_match_counter = np.pad(self.finished_match_counter, (0, 1),
                                           'constant')
      self.all_match_counter = np.pad(self.all_match_counter, (0, 1),
                                      'constant')
      self.sum_outcome = np.pad(self.sum_outcome, (0, 1), 'constant')
    for player in self.players:
      self.finished_matches[(p, player)] = []
      self.finished_matches[(player, p)] = []
      self.ongoing_matches[(p, player)] = {}
      self.ongoing_matches[(player, p)] = {}

  def start_match(self, rowp, colp, match_id):
    """ Record a started match.

    Typically called on actor task beginning.

    Args:
      rowp: row player
      colp: column player
      match_id:
    """
    i, j = self.pair2idx(rowp, colp)
    self.ongoing_matches[(rowp, colp)][match_id] = time()
    self.all_match_counter[i, j] += 1

  def finish_match(self, rowp, colp, outcome, info, match_id):
    """ Record a finished match.

    Typically called on a match ending.

    Args:
      rowp: row player
      colp: column player
      outcome: +1 (win), -1 (lose), 0 (draw), None (early abort)
      info: dict, information
      match_id:
    """

    def _decay(z):
      return z if self._stat_decay is None else self._stat_decay * z

    assert outcome in [-1, 0, 1, None]  # None means early abort.
    i, j = self.pair2idx(rowp, colp)
    if outcome is None:
      self.ongoing_matches[(rowp, colp)].pop(match_id)
      self.all_match_counter[i, j] -= 1
      return

    self.sum_outcome[i, j] = _decay(self.sum_outcome[i, j])
    self.sum_outcome[j, i] = _decay(self.sum_outcome[j, i])
    self.sum_outcome[i, j] += outcome
    # NOTE: doesn't need update sum_outcome[j, i]

    self.finished_match_counter[i, j] = _decay(
      self.finished_match_counter[i, j]
    )
    self.finished_match_counter[j, i] = _decay(
      self.finished_match_counter[j, i]
    )
    self.finished_match_counter[i, j] += 1
    # NOTE: doesn't need update finished_match_counter[j, i]

    self.finished_matches[(rowp, colp)].append(outcome)
    start_time = self.ongoing_matches[(rowp, colp)].pop(match_id)
    if self.avg_match_time is None:
      self.avg_match_time = time() - start_time
    else:
      self.avg_match_time = 0.9 * self.avg_match_time + 0.1 * (
            time() - start_time)
    # Write to file
    if self.pgn_file:
      self.write_pgn(rowp, colp, outcome, match_id, info)

  def write_pgn(self, rowp, colp, outcome, match_id='', info=None):
    info = info or {}
    assert rowp in self.players
    assert colp in self.players
    assert outcome in [-1, 0, 1]
    s = ''
    s += '[Event "%s"]\n' % str(match_id)
    s += '[Site "%s"]\n' % str(match_id)
    s += '[Date "%s"]\n' % datetime.now().strftime("%Y-%m-%d %H:%M:%OS")
    s += '[Round "%s"]\n' % 'round'
    s += '[White "%s"]\n' % str(rowp)
    s += '[Black "%s"]\n' % str(colp)
    result = '1/2-1/2'
    if outcome == 1:
      result = '1-0'
    elif outcome == -1:
      result = '0-1'
    elif outcome == 0:
      result = '1/2-1/2'
    else:
      raise RuntimeError("Unknown outcome")
    s += '[Result "%s"]\n' % result
    s += '\nInfo: %s\n\n' % str(info)
    self.pgn_file.write(s)
    self.pgn_file.flush()

  def get_payoff_matrix(self):
    """ Get Payoff Matrix. Count based statistics """
    A = make_payoff_matrix(self.sum_outcome, self.finished_match_counter)
    logger.log("Payoff Matrix:", level=logger.DEBUG)
    logger.log(A, level=logger.DEBUG)
    return A

  def get_one_vs_all_info(self, cur_player, other_players, minimal_matches=10):
    """Get one-vs-all matche info, i.e., a 1 x M row slice of the payoff matrix.

    Args:
      cur_player: current player, the row of the payoff matrix
      other_players: other players, the columns of the payoff matrix
      minimal_matches: if <= this number, the winrate is deemed as 0.5 due to
        low confidence

    Returns:
      A winrate vector, np.array.
      A sum outcome vector, np.array.
      A match count vector, np.array.
    """
    # row vector (1, N) -> (N,)
    ind_row = [self.players.index(cur_player)]
    ind_cols = [self.players.index(p) for p in other_players]
    r_sum_outcome = self.sum_outcome[np.ix_(ind_row, ind_cols)].squeeze(axis=0)
    r_match_count = (
      self.finished_match_counter[np.ix_(ind_row, ind_cols)].squeeze(axis=0)
    )
    # column vector (N, 1) -> (N,)
    ind_rows = [self.players.index(p) for p in other_players]
    ind_col = [self.players.index(cur_player)]
    c_sum_outcome = self.sum_outcome[np.ix_(ind_rows, ind_col)].squeeze(axis=1)
    c_match_count = (
      self.finished_match_counter[np.ix_(ind_rows, ind_col)].squeeze(axis=1)
    )
    # enforce anti-symmetry (!!) by taking average
    sum_outcome = r_sum_outcome - c_sum_outcome
    match_count = r_match_count + c_match_count
    # outcome [-1, +1} -> winrate [0, 1]
    winrate = (1.0 + sum_outcome / (match_count + 0.000001)) / 2.0
    # dismiss the low-confident matches
    dft_wr = 0.5
    winrate[match_count < minimal_matches] = dft_wr
    return winrate, sum_outcome, match_count

  def get_player(self, current_player):
    """ Get a new player given `current_player`.

    Typically used for a leaner task. Player as alias of model_key.
    The returned new player is from the player pool, indicating whether to
    continue training `current_player` or inheriting from other player.

    Args:
      current_player: current player (model key).

    Returns:
      A new player (model key), which can be:
      * None, indicating to continue training with the same model key as
        `current_player`.
      * `current_player`, indicating to continue training but with a new model
         key.
      * other model key.
    """
    raise NotImplemented

  def get_opponent(self, player, hyperparam):
    """ Get an opponent for `player` using `hyperparam`.

    Typically used for an actor task. Return a player selected from the player
    pool as the opponent for `player`. It implements the opponent match making
    algorithm.

    Args:
      player: player (model key)
      hyperparam: hyper-parameters

    Returns:
      A new player as the opponent for (model key).
    """
    raise NotImplemented

  def get_eval_match(self):
    """ Get player pair for a match evaluation. Typically for an eval actor
    task. Return player model_key1, model_key2 """
    raise NotImplemented
