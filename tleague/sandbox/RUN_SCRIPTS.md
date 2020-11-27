## Install
Follow the steps below:

* cd to `TLeague` folder and run:
```bash
pip install -e .
```
(Almost) All `py` dependencies will be installed. 
Note, we've intentionally omitted the following packages that you need manually install:
```bash
tensorflow  # pip install tensorflow or tensorflow-gpu or cutomized *.whl
arena  # internal package at this moment, use pip install -e .
pysc2  # Use Tencent extension, use pip install -e . 
``` 

* Install `gambit` (for calculating Nash Equilibrium) on the machine where `league_mgr` deploys.
See the [link](https://gambitproject.readthedocs.io/en/latest/build.html#general-information) here.
Note: currently only the binary is needed, 
you don't have to install the `python` extension. 

* Install the following binaries by `yum` or `apt-get` or `homebrew`:
```bash
tmux
```

## Run from Commandline with Scripts

Two ways to start the training pipeline:

### Start each worker(role) manually
Run each of the worker commands in a seperate terminal/tmux,
either on a single machine or across different machines.
Only the learner uses a GPU.

Example 1:
```
# actor
CUDA_VISIBLE_DEVICES= python -m tleague.scripts.run_ppo_actor \
    --league_mgr_addr localhost:10005 \
    --model_pool_addrs localhost:10003:10004 \
    --learner_addr localhost:10001:10002

# actor eval
CUDA_VISIBLE_DEVICES= python -m tleague.scripts.run_ppo_actor_eval \
    --league_mgr_addr localhost:10005 \
    --model_pool_addrs localhost:10003:10004 

# learner
CUDA_VISIBLE_DEVICES=0 python -m tleague.scripts.run_ppo_learner \
    --league_mgr_addr localhost:10005 \
    --model_pool_addrs localhost:10003:10004 \
    --learner_ports 10001:10002

# league mgr
CUDA_VISIBLE_DEVICES= python -m tleague.scripts.run_league_mgr \
    --model_pool_addrs localhost:10003:10004 \
    --port 10005

# model pool
CUDA_VISIBLE_DEVICES= python -m tleague.scripts.run_model_pool \
    --ports 10003:10004
```

### Use a high-level frontend script
Run the frontend script which calls other scricpts in tmux windows.
For local machine, it simply runs the command. 
For remote machine, it performs connecting, copying the code and pip install before running the command.

Example 1.
Run all on local machines:
```bash
python -m tleague.scripts.run_ppo_all \
  --clust_spec_csv_path /Users/pengsun/code/TLeague/tleague/sandbox/clust_spec_example.csv \
  --tmux_sess tlea \
  --local_worker_pre_cmd "source /Users/pengsun/miniconda3/etc/profile.d/conda.sh && conda activate tlea && conda deactivate && conda activate tlea && "
```
where the `clust_spec_example.csv` specifies only localhosts.
Note: activating conda env seems problematic, 
it only works by activating-deactivating several times,
don't know why...

Example 2.
Run across both local and remote machines.
```bash
python -m tleague.scripts.run_ppo_all \
  --n_process 8 \
  --clust_spec_csv_path ../sandbox/clust_spec_example4.csv \
  --tmux_sess tlea \
  --remote_working_folder /data1/pythonsun \
  --local_worker_pre_cmd "source /data1/pythonsun/venv/tlea/bin/activate && " \
  --remote_worker_pre_cmd "source /home/work/tmp_pythonsun/venv/tlea/bin/activate && "
```
see `../sandbox/clust_spec_example2.csv`.
Use up to `n_process` parallel threads, 
i.e., connect to `n_process` machines (including localhost) simultaneously.

Example 3. First prepare on remote, then run the real commands:
```bash
python -m tleague.scripts.prepare_all \
  --n_process 8 \
  --clust_spec_csv_path ../sandbox/clust_spec_example4.csv \
  --remote_working_folder /home/work \
  --remote_worker_pre_cmd "source /home/work/tmp_pythonsun/venv/tlea/bin/activate && " \
  -p /local/root/Arena,/remote/root/Arena \
  -p /local/somepath/TencentPySC2,/remote/root/TencentPySC2 \
  -p /some/other/pac,/remote/pac
```
The script copies self folder `TLeague` to remote and do `pip install -e .`.
Then, for EACH of the package specified by `-p` args, 
it copies the local folder to remote folder (separated by comma) and to `pip install -e .`.

Then run the real commands:
```bash
python -m tleague.scripts.run_ppo_all \
  --noforce_overwrite_remote \
  --n_process 8 \
  --clust_spec_csv_path ../sandbox/clust_spec_example4.csv \
  --tmux_sess tlea \
  --remote_working_folder /home/work \
  --local_worker_pre_cmd "export http_proxy= && export https_proxy= && source /data1/pythonsun/venv/tlea/bin/activate && " \
  --remote_worker_pre_cmd "export http_proxy= && export https_proxy= && source /home/work/tmp_pythonsun/venv/tlea/bin/activate && "
```
Note the `--noforce_overwrite_remote` arg, 
as we've prepared everything in previous step. 
This agr saves you from doing unnecessary remote preparation when you run a new experiment
, if you're sure the remote code needs no updating or re-installing. doesnt need any preparation. 


#### The CSV file format
job, ip, port1, port2, cuda_visisble_devices, ssh_username, ssh_password, ssh_port

Note: `cuda_visisble_devices` will be added as ENV VAR prefix to each python call,
see `run_ppo_all.py`.

Note: `run_ppo_actor` does not require a `port` for the actor itself.
However, the frontend script requires a `port` for each actor (see the example CSV file) 
to distinguish each actor (using `ip:port` as ID) when paring the actor-learner.

#### Kill all processes & tmux session on each machine
```bash
python -m tleague.scripts.kill_all \
  --n_process 8 \
  --clust_spec_csv_path /Users/pengsun/code/TLeague/tleague/sandbox/clust_spec_example.csv \
  --tmux_sess tlea
```

#### Run Customized Commands on each machine
Run customized command (pip install, kill -9 wild processes, rm -rf undesired folder, etc.) on each machine:
```bash
python -m tleague.scripts.exec_cmd_remote --n_process 8 \
  --clust_spec_csv_path /Users/pengsun/code/TLeague/tleague/sandbox/clust_spec_example.csv \
  -r "ps -ef | grep StarCraft | grep -v grep | awk '{print \$2}' | xargs kill -9" \
  -r "ps -ef | grep vizdoom | grep -v grep | awk '{print \$2}' | xargs kill -9" \
  -r "cd /home/work/Arena && pip install -e ."
```
NOTE: Use the quote " and the backslash \ to make the command intact.
NOTE: it's extremely dangerous, for your own consideration!

#### Convert c.oa CSV 
Currently, the downloaded coa files are in `*.xlsx` format and can be more than one from a single order. 
You can place all the `*.xlsx` files in a folder and do the converting: 
```bash
export COA_FOLDER="coa_folder"
ls ${COA_FOLDER}/*.xlsx | xargs ./xlsx2csv.py > ${COA_FOLDER}/all.csv
python3 -m tleague.scripts.convert_coa \
  --input_path ${COA_FOLDER}/all.csv \
  --ssh_username root \
  --ssh_password Server@AI2017 \
  --base_port 20001 \
  --output_path clust_spec.csv
```
where the `xlsx2csv` combines several `*.xlsx` files and outputs a single `*.csv` file.
Then the convert_coa converts it to our clust_spec CSV.

