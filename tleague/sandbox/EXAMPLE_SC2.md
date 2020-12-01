# learner

python3 -m tleague.scripts.run_ppo_learner2 \
 --learner_id lrn0 \
 --env sc2full_formal5_dict \
 --policy tpolicies.net_zoo.mnet_v5.mnet_v5d2 \
 --rm_size 32 \
 --batch_size 8 \
 --total_timesteps 5000000 \
 --policy_config "{'use_xla': True, 'test': False, 'use_loss_type': 'rl', 'use_value_head': True, 'use_self_fed_heads': False, 'use_lstm': True, 'nlstm': 256, 'hs_len': 256*2, 'lstm_duration': 1, 'lstm_dropout_rate': 0.0, 'use_base_mask': True, 'lstm_cell_type': 'lstm', 'lstm_layer_norm': True, 'weight_decay': 0.00002, 'arg_scope_type': 'type_b', 'endpoints_verbosity': 10, 'n_v': 5, 'distillation': True, 'fix_all_embed': False, 'zstat_embed_version': 'v2'}" \
 --learner_config "{'vf_coef': 0.5, 'max_grad_norm': 1.0, 'distill_coef': [0.0005, 0.0005, 0.0002, 0.0002, 0.0002, 0.0004, 0.00067, 0.0003, 0.002, 0.00033, 0.0005], 'ent_coef': [0.00005, 0.00005, 0.00002, 0.00002, 0.00002, 0.00004, 0.000067, 0.00003, 0.0002, 0.000033, 0.00005],}" \
 --rwd_shape \
 --rollout_length 8 

# actor

python3 -m tleague.scripts.run_ppo_actor \
 --env sc2full_formal5_dict \
 --policy tpolicies.net_zoo.mnet_v5.mnet_v5d2 \
 --policy_config "{'use_xla': True, 'test': False, 'use_loss_type': 'rl', 'use_value_head': True, 'use_self_fed_heads': False, 'use_lstm': True, 'nlstm': 256, 'hs_len': 256*2, 'lstm_duration': 1, 'lstm_dropout_rate': 0.0, 'use_base_mask': True, 'lstm_cell_type': 'lstm', 'lstm_layer_norm': True, 'weight_decay': 0.00002, 'arg_scope_type': 'type_b', 'endpoints_verbosity': 10, 'n_v': 5, 'distillation': False, 'fix_all_embed': False, 'zstat_embed_version': 'v2'}" \
 --n_v 5 \
 --rwd_shape \
 --distillation \
 --interface_config "{'zstat_data_src': '/Users/pengsun/code/sc2_rl/TLeague/tleague/sandbox/tmp_rp1429-mv-zstat-tmp-selected2', 'mmr': 7000, 'max_bo_count': 50}" \
 --agent tleague.actors.agent.PPOAgent2 \
 --verbose 11


# league mgr

python3 -m tleague.scripts.run_league_mgr \
 --mutable_hyperparam_type DiscreteDistribHyperparam \
 --hyperparam_config_name "{'learning_rate': 0.00001, 'lam': 0.99, 'gamma': 1.0, 'reward_weights': [[1.0, 0.1, 0.1, 0.1, 0.1]], 'distill_model_key': 'None:init_model', 'n': 4, 'zeroing_prob': 0.2, 'lrn_id_to_distrib_type': {'lrn0': 'uniform'}}" \
 --game_mgr_type AEMatchMakingGameMgr \
 --init_model_path "/Users/pengsun/code/sc2_rl/TLeague/tleague/sandbox/tmp_init_model/IL-model_20200302042629.model_value.pkl" \
 --verbose 9

# model pool

python3 -m tleague.scripts.run_model_pool \
  --verbose 0