from absl import app
import numpy as np

from tleague.game_mgr.ae_game_mgrs import AEMatchMakingGameMgr, \
  AEMatchMakingGameMgrV2, NGUMatchMakingGameMgr, IndivMainAgent,\
  IndivMainExploiter, IndivLeagueExploiter, IndivCyclicExploiter,\
  IndivSKMainAgent, IndivEvoExploiter, IndivAdaEvoExploiter


def check_save_load(gm, gm_cls):
  checkpoint_dir = 'tmp_checkpoint_test'
  gm.save(checkpoint_dir)
  # write a pseudo filename.list
  with open(checkpoint_dir + '/filename.list', 'w') as f:
    filenames = [p + '_20000101010101.model\n' for p in gm.players]
    f.writelines(filenames)
  gm_new = gm_cls(verbose=0)
  gm_new.load(checkpoint_dir)
  assert gm.players == gm_new.players
  assert gm.finished_matches == gm_new.finished_matches
  assert np.sum(np.abs(gm.finished_match_counter - gm_new.finished_match_counter)) == 0
  assert np.sum(np.abs(gm.sum_outcome - gm_new.sum_outcome)) == 0
  assert gm._population.keys() == gm_new._population.keys()
  if (hasattr(gm, '_active_role_count')
      and hasattr(gm_new, '_active_role_count')):
    assert gm._active_role_count == gm_new._active_role_count


def AEMatchMakingGameMgr_detailed_test():

  lrn_id_list = ['lrngrp0', 'lrngrp1', 'lrngrp2']

  gm = AEMatchMakingGameMgr(
    verbose=0,
    lrn_id_list=lrn_id_list,
    main_agent_pfsp_prob=0.7,
    main_agent_forgotten_prob=0.2,
    main_agent_forgotten_me_winrate_thre=0.5,
    main_agent_forgotten_ma_winrate_thre=0.7,
    main_exploiter_min_lps=3,
    main_exploiter_max_lps=4,
    main_exploiter_winrate_thre=0.1,
    main_exploiter_reset_winrate_thre=0.7,
    main_exploiter_prob_thre=0.15,
    league_exploiter_min_lps=6,
    league_exploiter_max_lps=7,
    league_exploiter_winrate_thre=0.7,
    league_exploiter_prob_thre=0.25,
  )

  # add initial model
  root = 'None:init_model'
  gm.add_player(p=root, parent_p=None)
  indiv = gm._population[root]
  assert isinstance(indiv, IndivMainAgent)
  assert indiv.is_historical is True
  assert indiv.total_trained_lps == 0

  # add Main Agent
  p1 = 'init_model:0001'
  gm.add_player(p1, root, learner_id='lrngrp0')
  indiv1 = gm._population[p1]
  assert isinstance(indiv1, IndivMainAgent)
  assert indiv1.is_historical is False
  assert indiv1.total_trained_lps == 0
  gm._dbg_coin_toss_oppo = 0.65  # for pfsp branch
  gm.get_opponent(p1, hyperparam=None)
  gm._dbg_coin_toss_oppo = 0.75  # for forgotten-opponent branch
  gm.get_opponent(p1, hyperparam=None)
  gm._dbg_coin_toss_oppo = 0.95  # for sp branch
  gm.get_opponent(p1, hyperparam=None)

  # add Main Exploiter
  p2 = 'init_model:0002'
  gm.add_player(p2, root, learner_id='lrngrp1')
  indiv2 = gm._population[p2]
  assert isinstance(indiv2, IndivMainExploiter)
  assert indiv2.is_historical is False
  assert indiv2.total_trained_lps == 0
  for i in range(3):
    gm.get_opponent(p2, hyperparam=None)

  # add League Exploiter
  p3 = 'init_model:0003'
  gm.add_player(p3, root,  learner_id='lrngrp2')
  indiv3 = gm._population[p3]
  assert isinstance(indiv3, IndivLeagueExploiter)
  assert indiv3.is_historical is False
  assert indiv3.total_trained_lps == 0
  for i in range(3):
    gm.get_opponent(p3, hyperparam=None)

  # inherit from the Main Agent
  assert gm._population[p1].is_historical is False
  p, _ = gm.get_player(p1)
  assert gm._population[p1].is_historical is True
  assert p == p1
  p4 = '0001:0004'
  gm.add_player(p4, parent_p=p1, learner_id='lrngrp0')
  indiv4 = gm._population[p4]
  assert isinstance(indiv4, IndivMainAgent)
  assert indiv4.total_trained_lps == 1
  gm._dbg_coin_toss_oppo = None
  for i in range(3):
    gm.get_opponent(p4, hyperparam=None)
  # the sampling branch
  gm._dbg_coin_toss_oppo = 0.65  # pfsp branch
  gm.get_opponent(p4, hyperparam=None)
  gm._dbg_coin_toss_oppo = 0.78  # forgotten branch
  gm.get_opponent(p4, hyperparam=None)
  gm._dbg_coin_toss_oppo = 0.92  # sp branch
  gm.get_opponent(p4, hyperparam=None)

  # inherit from the Main Exploiter
  assert gm._population[p2].is_historical is False
  p, _ = gm.get_player(p2)
  assert gm._population[p2].is_historical is True
  p5 = '0002:0005'
  gm.add_player(p5, parent_p=p2, learner_id='lrngrp1')
  indiv5 = gm._population[p5]
  assert isinstance(indiv5, IndivMainExploiter)
  assert indiv5.total_trained_lps == 1
  gm._dbg_coin_toss_oppo = None
  for i in range(3):
    gm.get_opponent(p5, hyperparam=None)

  # inherit from the League Exploiter
  assert gm._population[p3].is_historical is False
  p, _ = gm.get_player(p3)
  assert gm._population[p3].is_historical is True
  p6 = '0003:0006'
  gm.add_player(p6, parent_p=p3, learner_id='lrngrp2')
  indiv6 = gm._population[p6]
  assert isinstance(indiv6, IndivLeagueExploiter)
  assert indiv6.total_trained_lps == 1
  gm._dbg_coin_toss_oppo = None
  for i in range(3):
    gm.get_opponent(p6, hyperparam=None)

  check_save_load(gm, AEMatchMakingGameMgr)


def AEMatchMakingGameMgr_main_agent_only_test():
  lrn_id_list = ['lrngrp0', 'lrngrp1', 'lrngrp2']

  gm = AEMatchMakingGameMgr(
    verbose=0,
    lrn_id_list=lrn_id_list,
    main_agent_pfsp_prob=0.7,
    main_agent_forgotten_prob=0.2,
    main_agent_forgotten_me_winrate_thre=0.5,
    main_agent_forgotten_ma_winrate_thre=0.7,
    main_exploiter_min_lps=3,
    main_exploiter_max_lps=4,
    main_exploiter_winrate_thre=0.1,
    main_exploiter_reset_winrate_thre=0.7,
    main_exploiter_prob_thre=0.15,
    league_exploiter_min_lps=6,
    league_exploiter_max_lps=7,
    league_exploiter_winrate_thre=0.7,
    league_exploiter_prob_thre=0.25,
  )

  # add initial model
  root = 'None:init_model'
  gm.add_player(p=root, parent_p=None, learner_id=None)
  indiv = gm._population[root]
  assert isinstance(indiv, IndivMainAgent)
  assert indiv.is_historical is True
  assert indiv.total_trained_lps == 0

  # add Main Agent
  p1 = 'init_model:0001'
  gm.add_player(p1, root, learner_id='lrngrp0')
  indiv1 = gm._population[p1]
  assert isinstance(indiv1, IndivMainAgent)
  assert indiv1.is_historical is False
  assert indiv1.total_trained_lps == 0
  gm._dbg_coin_toss_oppo = 0.42
  gm.get_opponent(p1, hyperparam=None)
  gm._dbg_coin_toss_oppo = 0.59
  gm.get_opponent(p1, hyperparam=None)
  gm._dbg_coin_toss_oppo = 0.72
  gm.get_opponent(p1, hyperparam=None)

  # inherit from the Main Agent
  assert gm._population[p1].is_historical is False
  p, _ = gm.get_player(p1)
  assert gm._population[p1].is_historical is True
  assert p == p1
  p2 = '0001:0002'
  gm.add_player(p2, parent_p=p1, learner_id='lrngrp0')
  indiv2 = gm._population[p2]
  assert isinstance(indiv2, IndivMainAgent)
  assert indiv2.total_trained_lps == 1
  gm._dbg_coin_toss_oppo = None
  for i in range(3):
    gm.get_opponent(p2, hyperparam=None)

  # inherit from the Main Agent
  assert gm._population[p2].is_historical is False
  p, _ = gm.get_player(p2)
  assert gm._population[p2].is_historical is True
  assert p == p2
  p3 = '0002:0003'
  gm.add_player(p3, parent_p=p2, learner_id='lrngrp0')
  indiv3 = gm._population[p3]
  assert isinstance(indiv3, IndivMainAgent)
  assert indiv3.total_trained_lps == 2
  gm._dbg_coin_toss_oppo = None
  for i in range(3):
    gm.get_opponent(p3, hyperparam=None)

  check_save_load(gm, AEMatchMakingGameMgr)


def AEMatchMakingGameMgr_main_exploiter_test():
  lrn_id_list = ['lrngrp0', 'lrngrp1', 'lrngrp2']
  main_exploiter_min_lps = 1
  main_exploiter_max_lps = 2

  gm = AEMatchMakingGameMgr(
    verbose=0,
    lrn_id_list=lrn_id_list,
    main_agent_pfsp_prob=0.7,
    main_agent_forgotten_prob=0.2,
    main_agent_forgotten_me_winrate_thre=0.5,
    main_agent_forgotten_ma_winrate_thre=0.7,
    main_exploiter_min_lps=main_exploiter_min_lps,
    main_exploiter_max_lps=main_exploiter_max_lps,
    main_exploiter_winrate_thre=0.1,
    main_exploiter_reset_winrate_thre=0.7,
    main_exploiter_prob_thre=0.15,
    league_exploiter_min_lps=6,
    league_exploiter_max_lps=7,
    league_exploiter_winrate_thre=0.7,
    league_exploiter_prob_thre=0.25,
  )

  # add initial model
  root = 'None:init_model'
  gm.add_player(p=root, parent_p=None, learner_id=None)
  indiv = gm._population[root]
  assert isinstance(indiv, IndivMainAgent)
  assert indiv.is_historical is True

  # add Main Agent
  p1 = 'init_model:0001'
  gm.add_player(p1, root, learner_id='lrngrp0')
  indiv1 = gm._population[p1]
  assert isinstance(indiv1, IndivMainAgent)
  assert indiv1.is_historical is False
  gm._dbg_coin_toss_oppo = 0.67
  gm.get_opponent(p1, hyperparam=None)
  gm._dbg_coin_toss_oppo = 0.79
  gm.get_opponent(p1, hyperparam=None)
  gm._dbg_coin_toss_oppo = 0.95
  gm.get_opponent(p1, hyperparam=None)

  # add Main Exploiter
  p2 = 'init_model:0002'
  gm.add_player(p2, root, learner_id='lrngrp1')
  indiv2 = gm._population[p2]
  assert isinstance(indiv2, IndivMainExploiter)
  assert indiv2.is_historical is False
  for i in range(3):
    gm.get_opponent(p2, hyperparam=None)

  # add League Exploiter
  p3 = 'init_model:0003'
  gm.add_player(p3, root, learner_id='lrngrp2')
  indiv3 = gm._population[p3]
  assert isinstance(indiv3, IndivLeagueExploiter)
  assert indiv3.is_historical is False
  for i in range(3):
    gm.get_opponent(p3, hyperparam=None)

  # inherit from the Main Exploiter
  assert gm._population[p2].is_historical is False
  p, _ = gm.get_player(p2)
  assert gm._population[p2].is_historical is True
  p5 = '0002:0005'
  gm.add_player(p5, parent_p=p2, learner_id='lrngrp1')
  indiv5 = gm._population[p5]
  assert isinstance(indiv5, IndivMainExploiter)
  assert indiv5.total_trained_lps == 1
  gm._dbg_coin_toss_oppo = None
  for i in range(3):
    gm.get_opponent(p5, hyperparam=None)

  # inherit from the Main Exploiter, reset to initial model
  assert gm._population[p5].is_historical is False
  p, _ = gm.get_player(p5)
  assert p == root
  assert gm._population[p5].is_historical is True
  p6 = '0005:0006'
  gm.add_player(p6, parent_p=root, learner_id='lrngrp1')
  indiv6 = gm._population[p6]
  assert isinstance(indiv6, IndivMainExploiter)
  assert indiv6.total_trained_lps == 0
  gm._dbg_coin_toss_oppo = None
  for i in range(3):
    gm.get_opponent(p6, hyperparam=None)

  check_save_load(gm, AEMatchMakingGameMgr)


def AEMatchMakingGameMgr_league_exploiter_test():
  lrn_id_list = ['lrngrp0', 'lrngrp1', 'lrngrp2']
  league_exploiter_min_lps = 1
  league_exploiter_max_lps = 2

  gm = AEMatchMakingGameMgr(
    verbose=0,
    lrn_id_list=lrn_id_list,
    main_agent_pfsp_prob=0.7,
    main_agent_forgotten_prob=0.2,
    main_agent_forgotten_me_winrate_thre=0.5,
    main_agent_forgotten_ma_winrate_thre=0.7,
    main_exploiter_min_lps=2,
    main_exploiter_max_lps=3,
    main_exploiter_winrate_thre=0.1,
    main_exploiter_reset_winrate_thre=0.7,
    main_exploiter_prob_thre=0.15,
    league_exploiter_min_lps=league_exploiter_min_lps,
    league_exploiter_max_lps=league_exploiter_max_lps,
    league_exploiter_winrate_thre=0.7,
    league_exploiter_prob_thre=0.25,
  )

  # add initial model
  root = 'None:init_model'
  gm.add_player(p=root, parent_p=None, learner_id=None)
  indiv = gm._population[root]
  assert isinstance(indiv, IndivMainAgent)
  assert indiv.is_historical is True

  # add Main Agent
  p1 = 'init_model:0001'
  gm.add_player(p1, root, learner_id='lrngrp0')
  indiv1 = gm._population[p1]
  assert isinstance(indiv1, IndivMainAgent)
  assert indiv1.is_historical is False
  gm._dbg_coin_toss_oppo = 0.65
  gm.get_opponent(p1, hyperparam=None)
  gm._dbg_coin_toss_oppo = 0.78
  gm.get_opponent(p1, hyperparam=None)
  gm._dbg_coin_toss_oppo = 0.92
  gm.get_opponent(p1, hyperparam=None)

  # add Main Exploiter
  p2 = 'init_model:0002'
  gm.add_player(p2, root, learner_id='lrngrp1')
  indiv2 = gm._population[p2]
  assert isinstance(indiv2, IndivMainExploiter)
  assert indiv2.is_historical is False
  for i in range(3):
    gm.get_opponent(p2, hyperparam=None)

  # add League Exploiter
  p3 = 'init_model:0003'
  gm.add_player(p3, root, learner_id='lrngrp2')
  indiv3 = gm._population[p3]
  assert isinstance(indiv3, IndivLeagueExploiter)
  assert indiv3.is_historical is False
  assert indiv3.total_trained_lps == 0
  for i in range(3):
    gm.get_opponent(p3, hyperparam=None)

  # inherit from the League Exploiter, continue training
  assert gm._population[p3].is_historical is False
  gm._dbg_coin_toss_reset = 0.015
  p, _ = gm.get_player(p3)
  assert gm._population[p3].is_historical is True
  assert p != root
  p4 = '0003:0004'
  gm.add_player(p4, parent_p=p3, learner_id='lrngrp2')
  indiv4 = gm._population[p4]
  assert isinstance(indiv4, IndivLeagueExploiter)
  assert indiv4.total_trained_lps == 1
  gm._dbg_coin_toss_oppo = None
  for i in range(3):
    gm.get_opponent(p4, hyperparam=None)

  # inherit from the League Exploiter, reset to initial model
  assert gm._population[p4].is_historical is False
  gm._dbg_coin_toss_reset = 0.015
  p, _ = gm.get_player(p4)
  assert gm._population[p4].is_historical is True
  assert p == root
  p5 = '0004:0005'
  gm.add_player(p5, parent_p=root, learner_id='lrngrp2')
  indiv5 = gm._population[p5]
  assert isinstance(indiv5, IndivLeagueExploiter)
  assert indiv5.total_trained_lps == 0
  gm._dbg_coin_toss_oppo = None
  for i in range(3):
    gm.get_opponent(p5, hyperparam=None)

  check_save_load(gm, AEMatchMakingGameMgr)


def AEMatchMakingGameMgrV2_main_exploiter_test():
  root = 'None:init_model'
  main_exploiter_min_lps = 2
  main_exploiter_max_lps = 3
  lrn_id_list = ['lrngrp0', 'lrngrp1', 'lrngrp2']
  gm = AEMatchMakingGameMgrV2(
    lrn_id_list=lrn_id_list,
    main_agent_pfsp_prob=0.5,
    main_agent_forgotten_prob=0.15,
    main_agent_forgotten_me_winrate_thre=0.3,
    main_agent_forgotten_ma_winrate_thre=0.7,
    main_exploiter_min_lps=main_exploiter_min_lps,
    main_exploiter_max_lps=main_exploiter_max_lps,
    main_exploiter_winrate_thre=0.05,
    main_exploiter_reset_winrate_thre=0.7,
    main_exploiter_prob_thre=0.15,
    league_exploiter_min_lps=1,
    league_exploiter_max_lps=2,
    league_exploiter_winrate_thre=0.05,
    league_exploiter_prob_thre=0.7,
    cyclic_exploiter_n_leaves=4,
    cyclic_exploiter_root_name=root,
    cyclic_exploiter_prob_thre=0.15,
    verbose=0
  )

  # add initial model
  gm.add_player(p=root, parent_p=None)
  indiv = gm._population[root]
  assert isinstance(indiv, IndivMainAgent)
  assert indiv.is_historical is True

  # add Main Agent
  p1 = 'init_model:0001'
  gm.add_player(p1, root, 'lrngrp0')
  indiv1 = gm._population[p1]
  assert isinstance(indiv1, IndivMainAgent)
  assert indiv1.is_historical is False
  gm._dbg_coin_toss_oppo = 0.42
  gm.get_opponent(p1, hyperparam=None)
  gm._dbg_coin_toss_oppo = 0.59
  gm.get_opponent(p1, hyperparam=None)
  gm._dbg_coin_toss_oppo = 0.72
  gm.get_opponent(p1, hyperparam=None)

  # add Main Exploiter
  p2 = 'init_model:0002'
  gm.add_player(p2, root, 'lrngrp1')
  indiv2 = gm._population[p2]
  assert isinstance(indiv2, IndivMainExploiter)
  assert indiv2.is_historical is False
  for i in range(3):
    gm.get_opponent(p2, hyperparam=None)

  # add Cyclic Exploiter
  p3 = 'init_model:0003'
  gm.add_player(p3, parent_p=root, learner_id='lrngrp2')
  indiv3 = gm._population[p3]
  assert isinstance(indiv3, IndivCyclicExploiter)
  assert indiv3.is_historical is False
  assert indiv3.total_trained_lps == 0
  for i in range(3):
    gm.get_opponent(p3, hyperparam=None)

  # begin testing
  def _update_model_key(mk):
    num = mk.split(':')[1]
    new_num = '%04d' % (int(num) + 1)
    return num + ':' + new_num

  def _model_key_num(mk):
    return int(mk.split(':')[1])

  current_player = p2
  for i in range(14):
    # decide inheritance
    new_player, _ = gm.get_player(current_player)

    # check if it is the correct inheritance
    if ((i + 1) % main_exploiter_max_lps) == 0:
      # should have reset model to init
      assert new_player == root
      pass
    else:
      # means continue training
      assert new_player == current_player
      pass

    #if i == 7:
    #  gm.save('sandbox/tmp_checkpoint_test')
    #  gm.load('sandbox/tmp_checkpoint_test')

    # update current player
    current_player = _update_model_key(current_player)
    gm.add_player(current_player, parent_p=new_player, learner_id='lrngrp1')

    # get opponents
    for j in range(4):
      o = gm.get_opponent(current_player, hyperparam=None)
      assert o == p1, 'oppo should be the active main agent'
      # print(o)

  check_save_load(gm, AEMatchMakingGameMgrV2)


def AEMatchMakingGameMgrV2_cyclic_exploiter_test():
  # cycle on this no. of exploiters
  lrn_id_list = ['lrngrp0', 'lrngrp1', 'lrngrp2']
  cyclic_exploiter_n_leaves = 4
  root = 'None:init_model'
  gm = AEMatchMakingGameMgrV2(
    lrn_id_list=lrn_id_list,
    main_agent_pfsp_prob=0.5,
    main_agent_forgotten_prob=0.15,
    main_agent_forgotten_me_winrate_thre=0.3,
    main_agent_forgotten_ma_winrate_thre=0.7,
    main_exploiter_min_lps=2,
    main_exploiter_max_lps=3,
    main_exploiter_winrate_thre=0.05,
    main_exploiter_reset_winrate_thre=0.7,
    main_exploiter_prob_thre=0.15,
    league_exploiter_min_lps=1,
    league_exploiter_max_lps=2,
    league_exploiter_winrate_thre=0.05,
    league_exploiter_prob_thre=0.7,
    cyclic_exploiter_n_leaves=cyclic_exploiter_n_leaves,
    cyclic_exploiter_root_name=root,
    cyclic_exploiter_prob_thre=0.15,  # to enforce beating active main agent
    verbose=0
  )

  # add initial model
  gm.add_player(p=root, parent_p=None)
  indiv = gm._population[root]
  assert isinstance(indiv, IndivMainAgent)
  assert indiv.is_historical is True

  # add Main Agent
  p1 = 'init_model:0001'
  gm.add_player(p1, root, 'lrngrp0')
  indiv1 = gm._population[p1]
  assert isinstance(indiv1, IndivMainAgent)
  assert indiv1.is_historical is False
  gm._dbg_coin_toss_oppo = 0.42
  gm.get_opponent(p1, hyperparam=None)
  gm._dbg_coin_toss_oppo = 0.59
  gm.get_opponent(p1, hyperparam=None)
  gm._dbg_coin_toss_oppo = 0.72
  gm.get_opponent(p1, hyperparam=None)

  # add Main Exploiter
  p2 = 'init_model:0002'
  gm.add_player(p2, root, 'lrngrp1')
  indiv2 = gm._population[p2]
  assert isinstance(indiv2, IndivMainExploiter)
  assert indiv2.is_historical is False
  for i in range(3):
    gm.get_opponent(p2, hyperparam=None)

  # add Cyclic Exploiter
  p3 = 'init_model:0003'
  gm.add_player(p3, parent_p=root, learner_id='lrngrp2')
  indiv3 = gm._population[p3]
  assert isinstance(indiv3, IndivCyclicExploiter)
  assert indiv3.is_historical is False
  assert indiv3.total_trained_lps == 0
  for i in range(3):
    gm.get_opponent(p3, hyperparam=None)

  # begin testing
  def _update_model_key(mk, pa_mk):
    num = mk.split(':')[1]
    new_num = '%04d' % (int(num) + 1)
    return pa_mk.split(':')[1] + ':' + new_num

  def _model_key(mk):
    return int(mk.split(':')[1])

  current_player = p3
  for i in range(14):
    # decide inheritance
    new_player, _ = gm.get_player(current_player)
    # check if it is the correct inheritance
    if i in [j for j in range(cyclic_exploiter_n_leaves - 1)]:
      assert new_player == root
    else:
      assert new_player != root
      parent_num = _model_key(new_player)
      current_num = _model_key(current_player)
      assert (parent_num + cyclic_exploiter_n_leaves - 1) == current_num

    # update current player
    #print('i {}, current_p {}, new_p {}'.format(i, current_player, new_player))
    current_player = _update_model_key(current_player, new_player)
    print('i {}, updated_current_p {}, parent_p {}'.format(
      i, current_player, new_player))
    gm.add_player(current_player, parent_p=new_player, learner_id='lrngrp2')

    #if i == 9:
    #  gm.save('sandbox/tmp_checkpoint_test2')
    #  gm.load('sandbox/tmp_checkpoint_test2')

    # get opponents
    for j in range(4):
      o = gm.get_opponent(current_player, hyperparam=None)
      assert o == p1, 'oppo should be the active main agent'
      # print(o)

  check_save_load(gm, AEMatchMakingGameMgrV2)


def SKMatchMakingGameMgr_main_agent_only_test():
  IndivMainAgent = IndivSKMainAgent
  lrn_id_list = ['lrngrp0', 'lrngrp1', 'lrngrp2']

  gm = NGUMatchMakingGameMgr(
    verbose=0,
    lrn_id_list=lrn_id_list,
    main_agent_pfsp_prob=0.7,
    main_agent_forgotten_prob=0.2,
    main_agent_forgotten_me_winrate_thre=0.5,
    main_agent_forgotten_ma_winrate_thre=0.7,
    main_agent_add_to_league_winrate_thre=0.4,
    main_exploiter_min_lps=3,
    main_exploiter_max_lps=4,
    main_exploiter_winrate_thre=0.1,
    main_exploiter_reset_winrate_thre=0.7,
    main_exploiter_prob_thre=0.15,
    league_exploiter_min_lps=6,
    league_exploiter_max_lps=7,
    league_exploiter_winrate_thre=0.7,
    league_exploiter_prob_thre=0.25,
  )

  # add initial model
  root = 'None:init_model'
  gm.add_player(p=root, parent_p=None, learner_id=None)
  print(gm._population)
  indiv = gm._population[root]
  print(type(indiv))
  print(IndivMainAgent)
  assert isinstance(indiv, IndivMainAgent)
  assert indiv.is_historical is True
  assert indiv.total_trained_lps == 0

  # add Main Agent
  p1 = 'init_model:0001'
  gm.add_player(p1, root, learner_id='lrngrp0')
  indiv1 = gm._population[p1]
  assert isinstance(indiv1, IndivMainAgent)
  assert indiv1.is_historical is False
  assert indiv1.total_trained_lps == 0
  gm._dbg_coin_toss_oppo = 0.42
  gm.get_opponent(p1, hyperparam=None)
  gm._dbg_coin_toss_oppo = 0.59
  gm.get_opponent(p1, hyperparam=None)
  gm._dbg_coin_toss_oppo = 0.72
  gm.get_opponent(p1, hyperparam=None)

  # inherit from the Main Agent
  assert gm._population[p1].is_historical is False
  p, _ = gm.get_player(p1)
  if p is None:
    # this indicates the win rate thres not satisfied yet
    assert gm._population[p1].is_historical is False
  else:
    # otherwise this reduces to the logic of main agent in AEMatchMakingGameMgr
    assert p == p1
    assert gm._population[p1].is_historical is True

    p2 = '0001:0002'
    gm.add_player(p2, parent_p=p1, learner_id='lrngrp0')
    indiv2 = gm._population[p2]
    assert isinstance(indiv2, IndivMainAgent)
    assert indiv2.total_trained_lps == 1
    gm._dbg_coin_toss_oppo = None
    for i in range(3):
      gm.get_opponent(p2, hyperparam=None)

    # inherit from the Main Agent
    assert gm._population[p2].is_historical is False
    p, _ = gm.get_player(p2)
    assert gm._population[p2].is_historical is True
    assert p == p2
    p3 = '0002:0003'
    gm.add_player(p3, parent_p=p2, learner_id='lrngrp0')
    indiv3 = gm._population[p3]
    assert isinstance(indiv3, IndivMainAgent)
    assert indiv3.total_trained_lps == 2
    gm._dbg_coin_toss_oppo = None
    for i in range(3):
      gm.get_opponent(p3, hyperparam=None)

  check_save_load(gm, NGUMatchMakingGameMgr)


def AEMatchMakingGameMgrV3_test():
  lrn_id_list = ['lrngrp0', 'lrngrp1', 'lrngrp2', 'lrngrp3']
  lrn_role_list = ['MA'] + ['ME'] * 3
  main_exploiter_min_lps = 1
  main_exploiter_max_lps = 1

  print(lrn_id_list, lrn_role_list)

  gm = AEMatchMakingGameMgr(
    verbose=0,
    lrn_id_list=lrn_id_list,
    lrn_role_list=lrn_role_list,
    main_agent_pfsp_prob=0.7,
    main_agent_forgotten_prob=0.2,
    main_agent_forgotten_me_winrate_thre=0.5,
    main_agent_forgotten_ma_winrate_thre=0.7,
    main_exploiter_min_lps=main_exploiter_min_lps,
    main_exploiter_max_lps=main_exploiter_max_lps,
    main_exploiter_winrate_thre=0.1,
    main_exploiter_reset_winrate_thre=0.7,
    main_exploiter_prob_thre=0.15,
    league_exploiter_min_lps=6,
    league_exploiter_max_lps=7,
    league_exploiter_winrate_thre=0.7,
    league_exploiter_prob_thre=0.25,
  )

  # add initial models, which are all treated as historical MA
  roots = ['None:init_model0', 'None:init_model1', 'None:init_model2', 'None:init_model3']
  for i in range(4):
    gm.add_player(p=roots[i], parent_p=None, learner_id=None)
    indiv = gm._population[roots[i]]
    assert isinstance(indiv, IndivMainAgent)
    assert indiv.is_historical is True

  # add Main Agent
  print('-'*60)
  p1 = 'init_model0:0001'
  print('Add main agent {}.'.format(p1))
  gm.add_player(p1, roots[0], learner_id='lrngrp0')
  indiv1 = gm._population[p1]
  assert isinstance(indiv1, IndivMainAgent)
  assert indiv1.is_historical is False
  gm._dbg_coin_toss_oppo = 0.67
  print('{}\'s opponents:'.format(p1))
  print(gm.get_opponent(p1, hyperparam=None))
  gm._dbg_coin_toss_oppo = 0.79
  print(gm.get_opponent(p1, hyperparam=None))
  gm._dbg_coin_toss_oppo = 0.95
  print(gm.get_opponent(p1, hyperparam=None))

  # add Main Exploiter
  print('-'*60)
  p2 = 'init_model1:0002'
  print('Add main exploiter {}.'.format(p2))
  gm.add_player(p2, roots[1], learner_id='lrngrp1')
  indiv2 = gm._population[p2]
  assert isinstance(indiv2, IndivMainExploiter)
  assert indiv2.is_historical is False
  print('{}\'s opponents:'.format(p2))
  for i in range(3):
    print(gm.get_opponent(p2, hyperparam=None))

  # add Main Exploiter
  print('-'*60)
  p3 = 'init_model2:0003'
  print('Add main exploiter {}.'.format(p3))
  gm.add_player(p3, roots[2], learner_id='lrngrp2')
  indiv3 = gm._population[p3]
  assert isinstance(indiv3, IndivMainExploiter)
  assert indiv3.is_historical is False
  print('{}\'s opponents:'.format(p3))
  for i in range(3):
    print(gm.get_opponent(p3, hyperparam=None))

  # add Main Exploiter
  print('-'*60)
  p4 = 'init_model3:0004'
  print('Add main exploiter {}.'.format(p4))
  gm.add_player(p4, roots[3], learner_id='lrngrp3')
  indiv4 = gm._population[p4]
  assert isinstance(indiv4, IndivMainExploiter)
  assert indiv4.is_historical is False
  print('{}\'s opponents:'.format(p4))
  for i in range(3):
    print(gm.get_opponent(p4, hyperparam=None))

  # inherit from the Main Agent
  print('-'*60)
  print('Inherit from main agent.')
  assert gm._population[p1].is_historical is False
  p, _ = gm.get_player(p1)
  assert gm._population[p1].is_historical is True
  assert p == p1
  p5 = '0001:0005'
  gm.add_player(p5, parent_p=p, learner_id='lrngrp0')
  indiv5 = gm._population[p5]
  assert isinstance(indiv5, IndivMainAgent)
  assert indiv5.total_trained_lps == 1
  gm._dbg_coin_toss_oppo = None
  print('{}\'s opponents:'.format(p5))
  for i in range(3):
    print(gm.get_opponent(p5, hyperparam=None))

  print('Now, main exploiters\' opponents are:')
  print(gm.get_opponent(p2, hyperparam=None))
  print(gm.get_opponent(p3, hyperparam=None))
  print(gm.get_opponent(p4, hyperparam=None))

  # inherit from the Main Exploiter
  print('-'*60)
  print('Inherit from main exploiter.')
  assert gm._population[p2].is_historical is False
  print('The main exploiter is {}'.format(p2))
  p, _ = gm.get_player(p2)
  print('After get_player is {}'.format(p))
  assert gm._population[p2].is_historical is True
  p6 = '0002:0006'
  gm.add_player(p6, parent_p=p2, learner_id='lrngrp1')
  indiv6 = gm._population[p6]
  assert isinstance(indiv6, IndivMainExploiter)
  assert indiv6.total_trained_lps == 1
  gm._dbg_coin_toss_oppo = None
  print('{}\'s opponents:'.format(p6))
  for i in range(3):
    print(gm.get_opponent(p6, hyperparam=None))

  # check_save_load(gm, AEMatchMakingGameMgr)


def AEMatchMakingGameMgr_EE_test():
  lrn_id_list = ['lrngrp0', 'lrngrp1']
  lrn_role_list = ['MA'] + ['EE-dz']
  main_exploiter_min_lps = 1
  main_exploiter_max_lps = 2

  gm = AEMatchMakingGameMgr(
    verbose=0,
    lrn_id_list=lrn_id_list,
    lrn_role_list=lrn_role_list,
    main_agent_pfsp_prob=0.7,
    main_agent_forgotten_prob=0.2,
    main_agent_forgotten_me_winrate_thre=0.5,
    main_agent_forgotten_ma_winrate_thre=0.7,
    main_exploiter_min_lps=main_exploiter_min_lps,
    main_exploiter_max_lps=main_exploiter_max_lps,
    main_exploiter_winrate_thre=0.1,
    main_exploiter_reset_winrate_thre=0.7,
    main_exploiter_prob_thre=0.15,
    spec_exploiter_min_lps=2,
    spec_exploiter_max_lps=30,
    spec_exploiter_winrate_thre=0.1,
    spec_exploiter_reset_winrate_thre=0.7,
    spec_exploiter_prob_thre=0.15,
    evo_exploiter_min_lps=1,
    evo_exploiter_max_lps=1,
    evo_exploiter_winrate_thre=0.1,
    evo_exploiter_reset_winrate_thre=0.7,
    evo_exploiter_prob_thre=0.15,
  )

  # add initial model
  root = 'None:init_model'
  gm.add_player(p=root, parent_p=None, learner_id=None)
  indiv = gm._population[root]
  assert isinstance(indiv, IndivMainAgent)
  assert indiv.is_historical is True

  # add Main Agent
  p1 = 'init_model:0001'
  gm.add_player(p1, root, learner_id='lrngrp0')
  indiv1 = gm._population[p1]
  assert isinstance(indiv1, IndivMainAgent)
  assert indiv1.is_historical is False
  gm._dbg_coin_toss_oppo = 0.67
  gm.get_opponent(p1, hyperparam=None)
  gm._dbg_coin_toss_oppo = 0.79
  gm.get_opponent(p1, hyperparam=None)
  gm._dbg_coin_toss_oppo = 0.95
  gm.get_opponent(p1, hyperparam=None)

  # add EE
  p2 = 'im2109-dz:0002'
  gm.add_player(p2, root, learner_id='lrngrp1')
  indiv2 = gm._population[p2]
  assert isinstance(indiv2, IndivEvoExploiter)
  assert indiv2.is_historical is False
  for i in range(3):
    gm.get_opponent(p2, hyperparam=None)

  # inherit from the Main Exploiter, reset to EE best
  assert gm._population[p2].is_historical is False
  p_next, _ = gm.get_player(p2)
  print(p_next)
  assert gm._population[p2].is_historical is True
  print(gm._population)

  p_new = '0002:0003'
  gm.add_player(p_new, parent_p=p_next, learner_id='lrngrp1')
  print(gm._population)
  indiv_new = gm._population[p_new]
  assert isinstance(indiv_new, IndivEvoExploiter)
  assert indiv_new.total_trained_lps == 0
  gm._dbg_coin_toss_oppo = None
  for i in range(3):
    print(gm.get_opponent(p_new, hyperparam=None))

  # check_save_load(gm, AEMatchMakingGameMgr)


def AEMatchMakingGameMgr_AEE_test():
  lrn_id_list = ['lrngrp0', 'lrngrp1']
  lrn_role_list = ['MA'] + ['AEE-dz']
  main_exploiter_min_lps = 1
  main_exploiter_max_lps = 2

  gm = AEMatchMakingGameMgr(
    verbose=0,
    lrn_id_list=lrn_id_list,
    lrn_role_list=lrn_role_list,
    main_agent_pfsp_prob=0.7,
    main_agent_forgotten_prob=0.2,
    main_agent_forgotten_me_winrate_thre=0.5,
    main_agent_forgotten_ma_winrate_thre=0.7,
    main_exploiter_min_lps=main_exploiter_min_lps,
    main_exploiter_max_lps=main_exploiter_max_lps,
    main_exploiter_winrate_thre=0.1,
    main_exploiter_reset_winrate_thre=0.7,
    main_exploiter_prob_thre=0.15,
    spec_exploiter_min_lps=2,
    spec_exploiter_max_lps=30,
    spec_exploiter_winrate_thre=0.1,
    spec_exploiter_reset_winrate_thre=0.7,
    spec_exploiter_prob_thre=0.15,
    evo_exploiter_min_lps=1,
    evo_exploiter_max_lps=1,
    evo_exploiter_winrate_thre=0.1,
    evo_exploiter_reset_winrate_thre=0.7,
    evo_exploiter_prob_thre=0.15,
    ada_evo_exploiter_min_lps=1,
    ada_evo_exploiter_max_lps=1,
    ada_evo_exploiter_winrate_thre=0.1,
    ada_evo_exploiter_reset_winrate_thre=0.7,
    ada_evo_exploiter_prob_thre=0.15,
  )

  # add initial model
  root = 'None:init_model'
  gm.add_player(p=root, parent_p=None, learner_id=None)
  indiv = gm._population[root]
  assert isinstance(indiv, IndivMainAgent)
  assert indiv.is_historical is True

  # add Main Agent
  p1 = 'init_model:0001'
  gm.add_player(p1, root, learner_id='lrngrp0')
  indiv1 = gm._population[p1]
  assert isinstance(indiv1, IndivMainAgent)
  assert indiv1.is_historical is False
  gm._dbg_coin_toss_oppo = 0.67
  gm.get_opponent(p1, hyperparam=None)
  gm._dbg_coin_toss_oppo = 0.79
  gm.get_opponent(p1, hyperparam=None)
  gm._dbg_coin_toss_oppo = 0.95
  gm.get_opponent(p1, hyperparam=None)

  # add EE
  p2 = 'init_model:0002'
  gm.add_player(p2, root, learner_id='lrngrp1')
  indiv2 = gm._population[p2]
  assert isinstance(indiv2, IndivAdaEvoExploiter)
  assert indiv2.is_historical is False
  for i in range(3):
    gm.get_opponent(p2, hyperparam=None)

  # inherit from the Main Exploiter, reset to EE best
  assert gm._population[p2].is_historical is False
  p_next, _ = gm.get_player(p2)
  print(p_next)
  assert gm._population[p2].is_historical is True
  print(gm._population)

  p_new = '0002:0003'
  gm.add_player(p_new, parent_p=p_next, learner_id='lrngrp1')
  print(gm._population)
  indiv_new = gm._population[p_new]
  assert isinstance(indiv_new, IndivAdaEvoExploiter)
  assert indiv_new.total_trained_lps == 0
  gm._dbg_coin_toss_oppo = None
  for i in range(3):
    print(gm.get_opponent(p_new, hyperparam=None))

  # check_save_load(gm, AEMatchMakingGameMgr)


def main(_):
  # # test AEMatchMakingGameMgr
  # AEMatchMakingGameMgr_detailed_test()
  # AEMatchMakingGameMgr_main_agent_only_test()
  # AEMatchMakingGameMgr_main_exploiter_test()
  # AEMatchMakingGameMgr_league_exploiter_test()
  #
  # # test AEMatchMakingGameMgrV2
  # AEMatchMakingGameMgrV2_main_exploiter_test()
  # AEMatchMakingGameMgrV2_cyclic_exploiter_test()
  #
  # # test SKMatchMakingGameMgr
  # SKMatchMakingGameMgr_main_agent_only_test()
  #
  # # test AEMatchMakingGameMgrV3
  # AEMatchMakingGameMgrV3_test()
  #
  # # test AEMatchMakingGameMgr_EE
  # AEMatchMakingGameMgr_EE_test()

  # test AEMatchMakingGameMgr_AEE
  AEMatchMakingGameMgr_AEE_test()


if __name__ == '__main__':
  app.run(main)
