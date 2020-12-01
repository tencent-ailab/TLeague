# Example for Running Gym Atari with PPO
`cd` to the directory `Tleague/examples` and run the following commandlines each in a separate terminal.
```Shell
bash example_gym_atari_ppo.sh model_pool
bash example_gym_atari_ppo.sh league_mgr
bash example_gym_atari_ppo.sh learner
bash example_gym_atari_ppo.sh actor
```

Single Agent RL can be viewed as a special case of MARL with `n_agents = 1`,
where most of the game manager logic is by-passed.
In this example we simply use a `tleague.game_mgr.game_mgrs.SelfPlayGameMgr` as a "placeholder".

We also use a large learning period length (`--total_timesteps=200000000`, 200M time steps) to prevent the model id from changing too frequently.
Note that training an NN all the way down suffices for most RL use cases. 