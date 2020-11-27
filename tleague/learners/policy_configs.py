#!/usr/bin/env python
""" Policy configs for different policies.

Suggested Convention:
PolicyName_Config_SuffixForGameAndOrVersion """


DeepMultiHeadMlpPolicy_Config_v1 = {
  'action_mask': True,
}

LstmDeepMHMlpPolicy_Config_v1 = {
  'action_mask': True,
  'rnn': True,
  'lstm': 128,
}

SpatialTrans_Config_sc2v1 = {
  'test': False,
  'rl': True,
}

PureConv_Config_sc2v1 = {
  'test': False,
  'rl': True,
}

# burn in sc2v1
PureConv_Config_sc2v1_1 = {
  'test': False,
  'rl': True,
  'fix_all_embed': True,
}

PureConv_Config_sc2v2 = {
  'test': False,
  'rl': True,
  'fix_ms_head': True,
}

PureConv_Config_sc2v3 = {
  'test': False,
  'rl': True,
  'fix_ms_head': True,
  'fix_loc_head': True,
}

PureConv_Config_sc2v4 = {
  'test': False,
  'rl': True,
  'fix_ability_head': False,
  'fix_noop_head': True,
  'fix_shift_head': True,
  'fix_tar_u_head': True,
  'fix_loc_head': True,
  'fix_ss_head': True,
  'fix_ms_head': True,
}

MNet_Config_sc2v1 = {
  'test': False,
  'rl': True,
  'use_noop_mask': True,
  'use_last_action': False,
}

MNet_Config_sc2v2 = {
  'test': False,
  'rl': True,
  'use_noop_mask': False,
  'use_last_action': True,
  'use_lstm': True,
  'nlstm': 64,
}

MNet_Config_IL_sc2v1 = {
  'test': False,
  'rl': False,
  'use_noop_mask': False,
  'use_last_action': True,
  'use_lstm': True,
  'nlstm': 64,
}

MNet_Config_IL_sc2v3 = {
  'test': False,
  'rl': False,
  'use_noop_mask': False,
  'use_last_action': True,
  'use_lstm': True,
  'nlstm': 64,
  'timi_version': 'v3'
}

MNet_Config_RL_sc2v3 = {
  'test': False,
  'rl': True,
  'use_noop_mask': False,
  'use_last_action': True,
  'use_lstm': True,
  'nlstm': 32,
  'timi_version': 'v3',
  'hs_len': 65,
  'lstm_duration': 8,
}

MNet_Config_IL_sc2v3_1 = {
  'test': False,
  'rl': False,
  'use_noop_mask': False,
  'use_last_action': False,
  'use_lstm': False,
  'timi_version': 'v3'
}

MNet_Config_IL_sc2v3_2 = {
  'test': False,
  'rl': False,
  'use_noop_mask': False,
  'use_last_action': True,
  'use_lstm': False,
  'timi_version': 'v3'
}

MNet_Config_IL_sc2v3_3 = {
  'test': False,
  'rl': False,
  'use_noop_mask': False,
  'use_last_action': False,
  'use_lstm': True,
  'nlstm': 64,
  'timi_version': 'v3'
}

MNetv4_Config_IL = {
  'test': False,
  'rl': False,
  'use_lstm': True,
  'nlstm': 256,
  'hs_len': 256*2+1,
}

MNetv5_Config_RL = {
  'use_xla': False,
  'test': False,
  'rl': True,
  'use_lstm': True,
  'nlstm': 256,
  'hs_len': 256*2,
  'lstm_duration': 1,
  'lstm_dropout_rate': 0.5,
  'use_base_mask': True,
  'lstm_cell_type': 'lstm',
  'lstm_layer_norm': True,
  'weight_decay': 0.00002,
  'arg_scope_type': 'type_b',
  'endpoints_verbosity': 10,
  'n_v': 5,
  'distillation': True
}