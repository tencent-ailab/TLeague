# Run from Commandline with Scripts

Example for pommerman:
```
# actor
python -m tleague.scripts.run_ppo_actor \
    --league_mgr_addr localhost:10005 \
    --model_pool_addrs localhost:10003:10004 \
    --learner_addr localhost:10001:10002 \
    --env pommerman --policy CnnPolicy

# actor eval
python -m tleague.scripts.run_ppo_actor_eval \
    --league_mgr_addr localhost:10005 \
    --model_pool_addrs localhost:10003:10004 \
    --env pommerman --policy CnnPolicy

# learner
python -m tleague.scripts.run_ppo_learner \
    --league_mgr_addr localhost:10005 \
    --model_pool_addrs localhost:10003:10004 \
    --learner_ports 10001:10002 \
    --gpu_id -1 \
    --env pommerman --policy CnnPolicy

# league mgr
python -m tleague.scripts.run_league_mgr \
    --model_pool_addrs localhost:10003:10004 \
    --port 10005 \
    --mutable_hyperparam_type HyperparamRandPredefPommerman

# model pool
python -m tleague.scripts.run_model_pool \
    --ports 10003:10004
```