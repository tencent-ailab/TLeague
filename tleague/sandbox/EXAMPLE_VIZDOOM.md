# Run from Commandline with Scripts

Example for vizdoom `dogfight`:
```
# model pool
python3 -m tleague.scripts.run_model_pool \
    --ports 10005:10006 \
    --verbose 0

# league mgr
python3 -m tleague.scripts.run_league_mgr \
    --port 10007 \
    --model_pool_addrs 127.0.0.1:10005:10006 \
    --game_mgr_type SelfPlayGameMgr \
    --mutable_hyperparam_type ConstantHyperparam \
    --hyperparam_config_name "{'learning_rate': 0.0001, 'lam': 0.99, 'gamma': 0.99, 'reward_weights': [[0.1, 0.1, 1.0, 0.1, 0.1, 0.3, 0.1, 0.3, 0.1, 0.005]]}" \
    --restore_checkpoint_dir "" \
    --save_checkpoint_root "/Users/pengsun/code/sc2_rl/TLeague/tleague/sandbox/tmp_vizdoom_dogfight_chkpoints" \
    --save_interval_secs 300 \
    --mute_actor_msg \
    --verbose 9 \
    --pseudo_learner_num -1

# actor
python3 -m tleague.scripts.run_ppo_actor \
    --league_mgr_addr 127.0.0.1:10007 \
    --model_pool_addrs 127.0.0.1:10005:10006 \
    --learner_addr 127.0.0.1:10001:10002 \
    --env vizdoom_dogfight \
    --unroll_length 32 \
    --update_model_freq 20 \
    --policy tpolicies.net_zoo.conv_lstm.conv_lstm \
    --policy_config "{'test': False, 'use_loss_type': 'rl', 'use_value_head': True, 'rollout_len': 1, 'use_lstm': True, 'nlstm': 64, 'hs_len': 128, 'lstm_dropout_rate': 0.2, 'lstm_layer_norm': True, 'weight_decay': 0.00002, 'sync_statistics': None}" \
    --verbose 11 \
    --log_interval_steps 51 \
    --n_v 1 \
    --rwd_shape \
    --nodistillation \
    --agent tleague.actors.agent.PPOAgent2 

# learner
python -m tleague.scripts.run_ppo_learner2 \
    --league_mgr_addr 127.0.0.1:10007 \
    --model_pool_addrs localhost:10005:10006 \
    --learner_ports 10001:10002 \
    --learner_id "lrn0" \
    --gpu_num 0 \
    --unroll_length 32 \
    --rollout_length 8 \
    --batch_size 16 \
    --rm_size 128 \
    --pub_interval 200 \
    --log_interval 100 \
    --total_timesteps 10000 \
    --burn_in_timesteps 0 \
    --env vizdoom_dogfight \
    --policy tpolicies.net_zoo.conv_lstm.conv_lstm \
    --policy_config "{'test': False, 'use_loss_type': 'rl', 'use_value_head': True, 'n_v': 1, 'use_lstm': True, 'nlstm': 64, 'hs_len': 128, 'lstm_dropout_rate': 0.2, 'lstm_layer_norm': True, 'weight_decay': 0.00002, 'sync_statistics': None}" \
    --learner_config "{'vf_coef': 0.5, 'ent_coef': 0.001, 'distill_coef': 0.0, 'max_grad_norm': 0.5}" \
    --norwd_shape
```

Note:
* `unroll_length` must be a multiple of `batch_size`. `unroll_length` means how long the trajectory is when computing the value using bootstrap.
* `rollout_length` = `rollout_len` in the `policy_config`. 