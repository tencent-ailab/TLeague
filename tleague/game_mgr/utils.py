#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 20190221 by qing
# 20190417 modified by pengsun (we don't actually need this, leave it to git)
import subprocess
import uuid
import tempfile
from os import path

import numpy as np
import scipy


def make_payoff_matrix(mat_sum_outcome, mat_match_count):
  """ make a payoff matrix. count based statistics, enforced symmetry.

  :param mat_sum_outcome: np.array, shape (N, N), sum of outcome
  :param mat_match_count: np.array, shape (N, N), count of matches
  :return: payoff matrix, np.array, shape (N, N)
  """
  A = ((mat_sum_outcome - mat_sum_outcome.transpose()) /
       (mat_match_count + mat_match_count.transpose() + 1e-8))
  return A


def mat2nfg(A, game_title='Title', players=['Player 1', 'Player 2']):
  assert len(A.shape) == 2
  assert len(players) == 2
  header = 'NFG 1 R "%s"' % game_title
  descrp = '{ "%s" "%s" } { %d %d }' % (
    players[0], players[1], A.shape[0], A.shape[1])
  body = ''
  for c in range(A.shape[1]):
    for r in range(A.shape[0]):
      body += '%f %f ' % (A[r][c], A[c][r])
  return '\n'.join([header, descrp, '', body])


def lcp_solve(A):
  import gambit
  gam = gambit.Game.parse_game(mat2nfg(A))
  res = gambit.nash.lcp_solve(gam)
  res = np.reshape(np.array(res, dtype=np.float32), (A.shape[0] + A.shape[1]))
  rv = res[0:A.shape[0]]
  cv = res[(A.shape[0]):(A.shape[0] + A.shape[1])]
  pmax = np.max(np.matmul(A, np.reshape(cv, (A.shape[1], 1))))
  pmin = np.min(np.matmul(np.reshape(rv, (1, A.shape[0])), A))
  # print("max: %e, min: %e" % (pmax, pmin))
  return rv, cv


def _get_gambit_gnm_path():
  p = '/home/work/ext/gambit/build/bin/gambit-gnm'  # default on our server
  # we use shell command which to decide the path, which should be fine if
  # gambit has been make install
  try:
    o = subprocess.check_output(['which', 'gambit-gnm'])
    p = o.decode('utf-8').rstrip('\n')
  except Exception as e:
    print("error while deciding gambit-gnm path {}:{} ".format(str(type(e)),
                                                               str(e)))
    print('using default gambit-gnm path')
  print('using gambit-gnm path {}'.format(p))
  return p


GAMBIT_GNM_PATH = _get_gambit_gnm_path()


def gnm_solve(A):
  def _safe_temp_nfg_filename():
    fn = str(uuid.uuid4())
    return path.join(tempfile.gettempdir(), '{}.nfg'.format(fn))

  assert len(A.shape) == 2
  nfg_file = _safe_temp_nfg_filename()
  with open(nfg_file, 'w') as f:
    f.write(mat2nfg(A))
  try:
    res = subprocess.check_output([GAMBIT_GNM_PATH, '-q', nfg_file])
    res = np.array(
      [float(each) for each in res.decode('utf-8').strip().split(',')[1:]],
      dtype=np.float32)
    rv = res[0:A.shape[0]]
    cv = res[(A.shape[0]):(A.shape[0] + A.shape[1])]
    pmax = np.max(np.matmul(A, np.reshape(cv, (A.shape[1], 1))))
    pmin = np.min(np.matmul(np.reshape(rv, (1, A.shape[0])), A))
    # print("max: %e, min: %e" % (pmax, pmin))
  except Exception as e:  # the procedure of finding nash equilibrium is fragile
    print("gnm_solve failed with exception '%s':%s " % (str(type(e)), str(e)))
    rv = np.array([1] * A.shape[0], dtype=np.float32) / A.shape[0]
    cv = np.array([1] * A.shape[1], dtype=np.float32) / A.shape[1]
    print("defaulting to equi-probabilities for row and column players.")
  return rv, cv


def elo_predict_winrate(r_row, r_col):
  """ Predict win-rate for the Row Player using Elo scores.
  See B.1 Algorithm 3, 4 of the paper
  Emergent Coordination Through Competition, ICLR2019

  :param r_row: rating of the row player
  :param r_col: rating of the column player
  :return: win-rate for the row player
  """
  # Be CAREFUL of the order of who-minus-who
  return 1.0 / (1 + 10 ** ((r_col - r_row) / 400.0))


def elo_update_rating(r_row, r_col, outcome, K=0.1):
  """ Update Elo ratings for a player pair given the game outcome.
  See B.3 Algorithm 4, Emergent Coordination Through Competition, ICLR2019
  :param r_row: rating of row player
  :param r_col: rating of column player
  :param outcome: game outcome, {+1, -1, 0}
  :param K: updating factor
  :return: updated_r_row, update_r_col
  """
  if outcome >= 0.999:
    s = +1.0
  elif outcome <= -0.999:
    s = 0.0
  else:
    s = 0.5

  s_elo = elo_predict_winrate(r_row, r_col)
  s_delta = s - s_elo
  updated_r_row = r_row + K * s_delta
  updated_r_col = r_col - K * s_delta
  return updated_r_row, updated_r_col


def winrates_to_prob(winrates, weighting="linear"):
  """Convert win-rates vector to probabilities (for, e.g., player sampling).

  Borrowed and modified from the pseudo code of AlphaStar Nature paper.
  """
  def variance_left_rectified(x, x_th):
    a = x * (1 - x)
    a[x < x_th] = x_th * (1 - x_th)
    return a

  winrates = (winrates if isinstance(winrates, np.ndarray)
              else np.asarray(winrates))
  weightings = {
    "variance": lambda x: x * (1 - x),
    "variance_lr0.1": lambda x: variance_left_rectified(x, x_th=0.1),
    "variance_lr0.05": lambda x: variance_left_rectified(x, x_th=0.05),
    "linear": lambda x: 1 - x,
    "linear_capped": lambda x: np.minimum(0.5, 1 - x),
    "squared": lambda x: (1 - x) ** 2,
  }
  fn = weightings[weighting]
  probs = fn(winrates)
  norm = probs.sum()
  if norm < 1e-10:
    return np.ones_like(winrates) / len(winrates)
  return probs / norm