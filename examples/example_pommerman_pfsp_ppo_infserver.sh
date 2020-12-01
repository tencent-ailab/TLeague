#!/bin/bash
# Example of sc2 multi-agent reinforcement learning without Inference Server, running on a single machine.
# Training from scratch with PPO algorithm, no teacher-student KL regularization (no distillation loss).

role=$1
# common args
game_mgr_type=tleague.game_mgr.ae_game_mgrs.AEMatchMakingGameMgr && \
game_mgr_config="{
  'lrn_id_list': ['lrngrp0'],
  'lrn_role_list': ['MA'],
  'main_agent_pfsp_prob': 0.5,
  'main_agent_forgotten_prob': 0.15,
  'main_agent_forgotten_me_winrate_thre': 0.5,
  'main_agent_forgotten_ma_winrate_thre': 0.7}"  # null for SelfPlayGameMgr
mutable_hyperparam_type=MutableHyperparam
hyperparam_config_name="{ \
  'learning_rate': 0.00001, \
  'lam': 0.8, \
  'gamma': 1.0, \
  'burn_in_timesteps': 10, \
  'reward_weights': [1.0, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], \
}" && \
policy=tpolicies.net_zoo.pommerman.conv_lstm;
policy_config="{ \
  'use_xla': False, \
  'test': False, \
  'rl': True, \
  'use_loss_type': 'rl', \
  'use_value_head': True, \
  'use_self_fed_heads': False, \
  'use_lstm': True, \
  'nlstm': 64, \
  'hs_len': 128, \
  'lstm_duration': 1, \
  'lstm_dropout_rate': 0.0, \
  'lstm_cell_type': 'lstm', \
  'lstm_layer_norm': True, \
  'weight_decay': 0.00000002, \
  'n_v': 11, \
  'merge_pi': False, \
}" && \
self_policy_config="{ \
  'batch_size': 1, \
  'rollout_len': 1, \
  'use_xla': False, \
  'test': True, \
  'use_loss_type': 'none', \
  'use_value_head': True, \
  'use_self_fed_heads': True, \
  'use_lstm': True, \
  'nlstm': 64, \
  'hs_len': 128, \
  'lstm_duration': 1, \
  'lstm_dropout_rate': 0.0, \
  'lstm_cell_type': 'lstm', \
  'lstm_layer_norm': True, \
  'weight_decay': 0.00000002, \
  'n_v': 11, \
  'merge_pi': False, \
}" && \
self_infserver_config="{ \
  'outputs': ['a', 'v', 'neglogp'], \
  'update_model_seconds': 30, \
  'model_key': '', \
}" && \
learner_config="{ \
  'vf_coef': [10, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], \
  'max_grad_norm': 1.0, \
  'distill_coef': 0, \
  'ent_coef': [0.01, 0.01] \
}" && \
env=pommerman_v2_fog && \
env_config="{ \
  'rotate': False, \
  'centralV': False, \
  'random_side': True \
}" && \
interface_config="{
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
  --unroll_length=32 \
  --rollout_length=8 \
  --batch_size=32 \
  --rm_size=64 \
  --pub_interval=100 \
  --log_interval=100 \
  --total_timesteps=2000000 \
  --burn_in_timesteps=12 \
  --env="${env}" \
  --policy="${policy}" \
  --policy_config="${policy_config}" \
  --batch_worker_num=1 \
  --rwd_shape \
  --learner_config="${learner_config}" \
  --type=PPO
fi

# self inference server
if [ $role == inf_server ]
then
python3 -m tleague.bin.run_inference_server \
  --port=30002 \
  --model_pool_addrs=localhost:10003:10004 \
  --league_mgr_addr=localhost:20005 \
  --learner_id="lrngrp0" \
  --env="${env}" \
  --is_rl \
  --policy="${policy}" \
  --policy_config="${self_policy_config}" \
  --infserver_config="${self_infserver_config}" \
  --batch_worker_num=1
fi

# actor
if [ $role == actor ]
then
python3 -m tleague.bin.run_pg_actor \
  --model_pool_addrs=localhost:10003:10004 \
  --league_mgr_addr=localhost:20005 \
  --learner_addr=localhost:30003:30004 \
  --self_infserver_addr=localhost:30002 \
  --unroll_length=32 \
  --update_model_freq=128 \
  --env="${env}" \
  --env_config="${env_config}" \
  --interface_config="${interface_config}" \
  --replay_dir=./tmp_trmmyy_replays \
  --policy="${policy}" \
  --policy_config="${self_policy_config}" \
  --log_interval_steps=3 \
  --n_v=11 \
  --norwd_shape \
  --nodistillation \
  --verbose=0 \
  --type=PPO
fi
