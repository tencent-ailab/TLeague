# View Logs
Each job/role will write log to stdout or some private log file.

`actor`: TODO

`learner`: TODO

`league_mgr`: Write `./example.pgn` and `./psro.log`. In particular,
any time you want during training you can do
```bash
bayeselo < rate.bayeselo
```
to generate the current Elo ranking for all the models in the pool.
The output file is named `elo.ratings` (see the script `rate.bayeselo`).
The extra binary TODO must have been installed.