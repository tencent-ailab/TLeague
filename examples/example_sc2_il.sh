#!/bin/bash
# Example of sc2 imitation learning without Inferenc Server, running on a single machine.

role=$1
# common args
replay_filelist=/Users/moe/testsc2/tmp/replays/tmp482.csv  # just a few replays from 4.8.2 game core
replay_dir=/Users/moe/testsc2/tmp/replays/rp1706-mv7-mmr6200-victory-selected-174-replays
zstat_data_src=/Users/moe/testsc2/tmp/replay_ds/rp1706-mv7-mmr6200-victory-selected-174
replay_converter="timitate.lib6.pb2all_converter.PB2AllConverter"
converter_config="{ \
  'zstat_data_src': '${zstat_data_src}', \
  'add_cargo_to_units': True, \
  'input_map_size': (128, 128), \
  'output_map_size': (128, 128), \
  'dict_space': True, \
  'sort_executors': 'v1', \
  'max_bo_count': 50, \
  'max_bobt_count': 20, \
  'zstat_zeroing_prob': 0.1, \
  'zmaker_version': 'v5', \
  'delete_dup_action': 'v2', \
  'inj_larv_rule': False, \
  'crop_to_playable_area': False, \
}"
policy=tpolicies.net_zoo.mnet_v6.mnet_v6d6
policy_config="{ \
  'use_xla': False, \
  'test': False, \
  'rl': False, \
  'use_lstm': True, \
  'nlstm': 384, \
  'hs_len': 384*2, \
  'lstm_duration': 1, \
  'lstm_dropout_rate': 0.1, \
  'use_base_mask': True, \
  'lstm_cell_type': 'lstm', \
  'lstm_layer_norm': True, \
  'weight_decay': 0.0000000001, \
  'arg_scope_type': 'mnet_v5_type_a', \
  'endpoints_verbosity': 10, \
  'use_self_fed_heads': False, \
  'use_loss_type': 'il', \
  'zstat_embed_version': 'v3', \
  'trans_version': 'v4', \
  'vec_embed_version': 'v3d1', \
  'embed_for_action_heads': 'lstm', \
  'use_astar_glu': True, \
  'use_astar_func_embed': True, \
  'pos_logits_mode': '1x1', \
  'pos_n_blk': 2, \
  'pos_n_skip': 2, \
}"

echo "Running as ${role}"

# model pool
if [ $role == model_pool ]
then
python3 -m tleague.bin.run_model_pool \
  --ports 10003:10004 \
  --verbose 0
fi

# learner
if [ $role == learner ]
then
python3 -m tleague.bin.run_imitation_learner3 \
  --model_pool_addrs=localhost:10003:10004 \
  --learner_spec=-1:30003:30004 \
  --replay_converter="${replay_converter}" \
  --converter_config="${converter_config}" \
  --replay_filelist="${replay_filelist}" \
  --checkpoints_dir=./tmp_imyymm_chkpoints \
  --learning_rate=0.001 \
  --min_train_sample_num=1 \
  --batch_size=2 \
  --rm_size=2 \
  --max_clip_grad_norm=1.0 \
  --print_interval=10 \
  --checkpoint_interval=50 \
  --min_val_sample_num=1 \
  --num_val_batches=2 \
  --policy="${policy}" \
  --policy_config="${policy_config}" \
  --unroll_length=2 \
  --rollout_length=2 \
  --pub_interval=10 \
  --train_generator_worker_num=1 \
  --repeat_training_task \
  --noenable_validation
fi

# actor
if [ $role == actor ]
then
python3 -m tleague.bin.run_replay_actor \
  --model_pool_addrs=localhost:10003:10004 \
  --learner_addr=localhost:30003:30004 \
  --replay_converter="${replay_converter}" \
  --converter_config="${converter_config}" \
  --replay_dir="${replay_dir}" \
  --step_mul=1 \
  --unroll_length=2 \
  --log_interval=5 \
  --update_model_freq=10 \
  --policy="${policy}" \
  --policy_config="${policy_config}" \
  --SC2_bin_root="/root" \
# Note for replay actor:
# * Do not pass --infserver_addr=host:port when not using Inference Server
# * --SC2_bin_root is used for Linux and is ignored for Darwin and Windows
fi
 