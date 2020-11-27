## Traditional self-play
TLeague can also be used to perform general self-play for a single learner with its historic versions as opponents.
Run each of the worker commands in a separate terminal/tmux,
either on a single machine or across different machines.
Only the learner uses a GPU.

Example for a battle game:
```
# actor
CUDA_VISIBLE_DEVICES= python -m tleague.scripts.run_ppo_actor \
    --league_mgr_addr localhost:10005 \
    --model_pool_addrs localhost:10003:10004 \
    --learner_addr localhost:10001:10002 \
    --env sc2_battle \
    --policy MiniTransformer
```

The PPOLearner allows a single learner preserve all the parameters, including these from the optimizer.
```
# learner
CUDA_VISIBLE_DEVICES=0 python -m tleague.scripts.run_ppo_learner \
    --league_mgr_addr localhost:10005 \
    --model_pool_addrs localhost:10003:10004 \
    --learner_ports 10001:10002 \
    --env sc2_battle \
    --policy MiniTransformer \
    --learner PPOLearner
```

HyperparamConstant uses the original reward. SelfPlayGameMgr returns the latest learner model and sample the recent top-10 historic versions.
```
# league mgr
CUDA_VISIBLE_DEVICES= python -m tleague.scripts.run_league_mgr \
    --model_pool_addrs localhost:10003:10004 \
    --port 10005 \
    --mutable_hyperparam_type HyperparamConstant \
    --game_mgr_type SelfPlayGameMgr
```

```
# model pool
CUDA_VISIBLE_DEVICES= python -m tleague.scripts.run_model_pool \
    --ports 10003:10004
```

## Nash Sampling during Training
Use the `RefCountGameMgr` or `PSROGameMgr`:
```bash
python -m tleague.scripts.run_ppo_all \
  --clust_spec_csv_path ../sandbox/clust_spec_example.csv \
  --tmux_sess tlea \
  --game_mgr_type RefCountGameMgr \
  --mutable_hyperparam_type MutableHyperparamRandPerturb \
  --hyperparam_config_name MutableHyperparamRandPerturb_SC2_v2 \
  --leagmgr_verbose 11 \
  --actor_verbose 11 \
  --local_worker_pre_cmd "source /Users/pengsun/miniconda3/etc/profile.d/conda.sh && conda activate tlea && conda deactivate && conda activate tlea && "
```

## PBT Elo Score Match Making
Use the `PBTEloMatchMakingGameMgr`:
```bash
python -m tleague.scripts.run_ppo_all \
  --clust_spec_csv_path ../sandbox/clust_spec_example.csv \
  --tmux_sess tlea \
  --game_mgr_type PBTEloMatchMakingGameMgr \
  --mutable_hyperparam_type MutableHyperparamRandPerturb \
  --hyperparam_config_name MutableHyperparamRandPerturb_SC2_v2 \
  --leagmgr_verbose 11 \
  --actor_verbose 11 \
  --local_worker_pre_cmd "source /Users/pengsun/miniconda3/etc/profile.d/conda.sh && conda activate tlea && conda deactivate && conda activate tlea && "
```
