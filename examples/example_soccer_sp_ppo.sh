#!/bin/bash
# Example of soccer multi-agent reinforcement learning without Inference Server, running on a single machine.
# Training from scratch with PPO algorithm, no teacher-student KL regularization (no distillation loss).

role=$1
# common args
game_mgr_type=tleague.game_mgr.game_mgrs.SelfPlayGameMgr && \
game_mgr_config="{
  'max_n_players': 30}"
mutable_hyperparam_type=ConstantHyperparam
hyperparam_config_name="{ \
  'learning_rate': 0.0001, \
  'lam': 0.99, \
  'gamma': 0.99, \
}" && \
policy=tpolicies.net_zoo.soccer.cont_nn;
policy_config="{ \
  'use_xla': False, \
  'test': False, \
  'rl': True, \
  'use_loss_type': 'rl', \
  'use_value_head': True, \
  'n_v': 1, \
  'use_lstm': False, \
}" && \
self_policy_config="{ \
  'batch_size': 32, \
  'rollout_len': 1, \
  'use_xla': False, \
  'test': True, \
  'use_loss_type': 'none', \
  'use_value_head': True, \
  'n_v': 1, \
  'use_lstm': False, \
  'sync_statistics': 'none', \
}" && \
learner_config="{ \
  'vf_coef': 0.5, \
  'max_grad_norm': 0.5, \
  'distill_coef': 0.0, \
  'ent_coef': 0.001 \
}" && \
env=soccer && \
env_config="{}" && \
interface_config="{}"

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
  --save_checkpoint_root=./tmp-trvd-yymmdd_chkpoints \
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
  --batch_size=32 \
  --rm_size=2 \
  --pub_interval=5 \
  --log_interval=4 \
  --total_timesteps=2000000 \
  --burn_in_timesteps=12 \
  --env="${env}" \
  --policy="${policy}" \
  --policy_config="${policy_config}" \
  --batch_worker_num=1 \
  --norwd_shape \
  --learner_config="${learner_config}" \
  --type=PPO
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
  --agent=tleague.actors.agent.PPOAgent2 \
  --policy="${policy}" \
  --policy_config="${self_policy_config}" \
  --log_interval_steps=3 \
  --n_v=1 \
  --rwd_shape \
  --nodistillation \
  --verbose=0 \
  --type=PPO
fi
