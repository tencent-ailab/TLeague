#!/bin/bash
# Example of sc2 multi-agent reinforcement learning without Inference Server, running on a single machine.
# Training from scratch with VTrace algorithm, no teacher-student KL regularization (no distillation loss).

role=$1
# common args
zstat_data_src=/Users/pengsun/code/tmp/replay_ds/rp1706-mv7-mmr6200-victory-selected-174
#zstat_data_src=/Users/jcxiong/SC2/rp1522-mv-zstat-tmp-selected-2
game_mgr_type=tleague.game_mgr.game_mgrs.SelfPlayGameMgr && \
game_mgr_config="{}"  # null for SelfPlayGameMgr
mutable_hyperparam_type=MutableHyperparam
hyperparam_config_name="{ \
  'learning_rate': 0.00001, \
  'lam': 0.8, \
  'gamma': 1.0, \
  'burn_in_timesteps': 10, \
  'reward_weights': [1.0, 1.0, 1.0, 1.5, 1.5, 1.5], \
}" && \
policy=tpolicies.net_zoo.mnet_v6.mnet_v6d6;
policy_config="{ \
  'use_xla': False, \
  'test': False, \
  'rollout_len': 2, \
  'rl': True, \
  'use_loss_type': 'rl_vtrace', \
  'use_value_head': True, \
  'use_self_fed_heads': False, \
  'use_lstm': True, \
  'nlstm': 384, \
  'hs_len': 768, \
  'lstm_duration': 1, \
  'lstm_dropout_rate': 0.0, \
  'lstm_cell_type': 'lstm', \
  'lstm_layer_norm': True, \
  'weight_decay': 0.00000002, \
  'arg_scope_type': 'mnet_v5_type_a', \
  'endpoints_verbosity': 10, \
  'n_v': 6, \
  'distillation': False, \
  'fix_all_embed': False, \
  'use_base_mask': True, \
  'zstat_embed_version': 'v3', \
  'trans_version': 'v4', \
  'vec_embed_version': 'v3d1', \
  'embed_for_action_heads': 'lstm', \
  'use_astar_glu': True, \
  'use_astar_func_embed': True, \
  'pos_logits_mode': '1x1', \
  'pos_n_blk': 2, \
  'pos_n_skip': 2, \
  'sync_statistics': 'none', \
  'temperature': 0.8, \
  'merge_pi': False, \
  'value_net_version': 'v2', \
}" && \
self_policy_config="{ \
  'batch_size': 2, \
  'rollout_len': 1, \
  'use_xla': False, \
  'test': True, \
  'use_loss_type': 'none', \
  'use_value_head': True, \
  'use_self_fed_heads': True, \
  'use_lstm': True, \
  'nlstm': 384, \
  'hs_len': 768, \
  'lstm_duration': 1, \
  'lstm_dropout_rate': 0.0, \
  'lstm_cell_type': 'lstm', \
  'lstm_layer_norm': True, \
  'weight_decay': 0.00000002, \
  'arg_scope_type': 'mnet_v5_type_a', \
  'endpoints_verbosity': 10, \
  'n_v': 6, \
  'distillation': False, \
  'fix_all_embed': False, \
  'use_base_mask': True, \
  'zstat_embed_version': 'v3', \
  'trans_version': 'v4', \
  'vec_embed_version': 'v3d1', \
  'embed_for_action_heads': 'lstm', \
  'use_astar_glu': True, \
  'use_astar_func_embed': True, \
  'pos_logits_mode': '1x1', \
  'pos_n_blk': 2, \
  'pos_n_skip': 2, \
  'sync_statistics': 'none', \
  'temperature': 0.8, \
  'merge_pi': False, \
  'value_net_version': 'v2', \
}" && \
learner_config="{ \
  'vf_coef': [5, 0.5, 0.5, 0.5, 0.5, 0.5], \
  'max_grad_norm': 1.0, \
  'distill_coef': 0, \
  'ent_coef': [0.00002, 0.00002, 0.00015, 0.00002, 0.00002, 0.00001], \
  'ep_loss_coef' : {'upgo_loss': 1} \
}" && \
env=sc2full_formal8_dict && \
env_config="{ \
  'use_trt': False, \
  'skip_noop': True, \
  'early_term': False, \
  'astar_rwd_version': 'v3' \
}" && \
interface_config="{ \
  'zstat_data_src': '${zstat_data_src}', \
  'mmr': 7000, \
  'max_bo_count': 50, \
  'max_bobt_count': 20, \
  'add_cargo_to_units': True, \
  'correct_pos_radius': 3.5, \
  'correct_building_pos': True, \
  'crop_to_playable_area': False, \
  'il_training': False, \
  'zstat_presort_order_name': '', \
  'zmaker_version': 'v5' \
}"

echo "Running as ${role}"

if [ $role == model_pool ]
then
# model pool
python3 -m tleague.bin.run_model_pool \
  --ports 10003:10004 \
  --verbose 0
fi

# league mgr
if [ $role == league_mgr ]
then
python3 -m tleague.bin.run_league_mgr \
  --port=20005 \
  --model_pool_addrs=localhost:10003:10004 \
  --game_mgr_type="${game_mgr_type}" \
  --game_mgr_config="${game_mgr_config}" \
  --mutable_hyperparam_type="${mutable_hyperparam_type}" \
  --hyperparam_config_name="${hyperparam_config_name}" \
  --restore_checkpoint_dir="" \
  --save_checkpoint_root=./tmp_trmmyy_chkpoints \
  --save_interval_secs=85 \
  --mute_actor_msg \
  --pseudo_learner_num=-1 \
  --verbose=0
fi

# learner
if [ $role == learner ]
then
python3 -m tleague.bin.run_pg_learner \
  --learner_spec=0:30003:30004 \
  --model_pool_addrs=localhost:10003:10004 \
  --league_mgr_addr=localhost:20005 \
  --learner_id=lrngrp0 \
  --unroll_length=2 \
  --rollout_length=2 \
  --batch_size=4 \
  --rm_size=2 \
  --pub_interval=5 \
  --log_interval=4 \
  --total_timesteps=2000000 \
  --burn_in_timesteps=12 \
  --env="${env}" \
  --policy="${policy}" \
  --policy_config="${policy_config}" \
  --batch_worker_num=1 \
  --rwd_shape \
  --learner_config="${learner_config}" \
  --type=Vtrace
fi

# actor
if [ $role == actor ]
then
python3 -m tleague.bin.run_pg_actor \
  --model_pool_addrs=localhost:10003:10004 \
  --league_mgr_addr=localhost:20005 \
  --learner_addr=localhost:30003:30004 \
  --unroll_length=2 \
  --update_model_freq=32 \
  --env="${env}" \
  --env_config="${env_config}" \
  --interface_config="${interface_config}" \
  --replay_dir=./tmp_trmmyy_replays \
  --agent=tleague.actors.agent.PGAgent \
  --policy="${policy}" \
  --policy_config="${self_policy_config}" \
  --log_interval_steps=3 \
  --n_v=6 \
  --norwd_shape \
  --nodistillation \
  --version=v2 \
  --verbose=0 \
  --type=Vtrace
fi
