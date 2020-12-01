python3 -m tleague.scripts.run_hvd_imitation_all \
  --n_process 1 \
  --clust_spec_csv_path clust_spec_hvd_peng.csv \
  --remote_working_folder /root/work \
  --noforce_overwrite_remote \
  --tmux_sess leo \
  --python_bin python3 \
  --prun horovodrun \
  --replay_converter=timitate.lib5.pb2all_converter.PB2AllConverter \
  --converter_config="{
    'zstat_data_src': '/root/replay_ds/rp1517-mv-zstat',
    'input_map_size': (128, 128),
    'output_map_size': (128, 128),
    'delete_useless_selection': False,
    'dict_space': True,
    'max_bo_count': 50,
    'max_bobt_count': 20,
    'zstat_zeroing_prob': 0.1,
    'zmaker_version': 'v4'
  }" \
  --replay_dir="/root/replays/ext_zvz" \
  --step_mul=1 \
  --unroll_length=32 \
  --log_interval=5 \
  --update_model_freq=64 \
  --policy=tpolicies.net_zoo.mnet_v5.mnet_v5d4 \
  --policy_config="{
    'use_xla': True,
    'test': False,
    'rl': False,
    'use_lstm': True,
    'nlstm': 256,
    'hs_len': 512,
    'lstm_duration': 1,
    'lstm_dropout_rate': 0.5,
    'use_base_mask': True,
    'lstm_cell_type': 'lstm',
    'lstm_layer_norm': True,
    'weight_decay': 1e-10,
    'arg_scope_type': 'type_a',
    'endpoints_verbosity': 10,
    'use_self_fed_heads': False,
    'use_loss_type': 'il',
    'zstat_embed_version': 'v3'
  }" \
  --modelpool_verbose=11 \
  --replay_filelist=/root/replays/471_480.filter.csv \
  --checkpoints_dir=/root/results/im1518_chkpoints \
  --learning_rate=0.001 \
  --min_train_sample_num=32 \
  --min_val_sample_num=32 \
  --batch_size=32 \
  --rm_size=256000 \
  --max_clip_grad_norm=0 \
  --print_interval=30 \
  --checkpoint_interval=2000 \
  --num_val_batches=200 \
  --rollout_length=16 \
  --pub_interval=100 \
  --train_generator_worker_num=6