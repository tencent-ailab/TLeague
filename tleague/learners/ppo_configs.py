#!/usr/bin/env python
""" Policy configs for different learner algorithm.

Suggested Convention:
LearnerName_Config_SuffixForGameAndOrVersion """


PPO_Config_v0 = {
  'ent_coef': 0.001,
  'vf_coef': 0.5,
  'max_grad_norm': 0.5,
  'distill_coef': 1e-2,
}

PPO_Config_mnetv0 = {
  'vf_coef': 0.5,
  'max_grad_norm': 0.5,
  'distill_coef': 0.0,
  'ent_coef':     0.001,
}

PPO_Config_mnetv1 = {
  'vf_coef': 0.5,
  'max_grad_norm': 0.5,
  'distill_coef': [5e-3, 1e-3, 1e-2, 1e-2, 1e-3, 1e-2, 1e-2, 1e-2, 1e-2, 1e-3, 1e-2],
  'ent_coef':     0.001,
}

PPO_Config_mnetv2 = {
  'vf_coef': 0.5,
  'max_grad_norm': 0.5,
  'distill_coef': [1e-3, 1e-3, 1e-2, 1e-2, 1e-3, 1e-2, 1e-2, 1e-2, 1e-2, 1e-3, 1e-2],
  'ent_coef':     0.001,
}

PPO_Config_mnetv3 = {
  'vf_coef': 0.5,
  'max_grad_norm': 0.5,
  'distill_coef': [5e-4, 1e-3, 1e-2, 1e-2, 1e-3, 1e-2, 1e-2, 1e-2, 1e-2, 1e-3, 1e-2],
  'ent_coef':     0.001,
}

PPO_Config_mnetv4 = {
  'vf_coef': 0.5,
  'max_grad_norm': 0.5,
  'distill_coef': [1e-4, 1e-3, 1e-2, 1e-2, 1e-3, 1e-2, 1e-2, 1e-2, 1e-2, 1e-3, 1e-2],
  'ent_coef':     0.001,
}

PPO_Config_mnetv5 = {
  'vf_coef': 0.5,
  'max_grad_norm': 0.5,
  'distill_coef': [5e-5, 1e-3, 1e-2, 1e-2, 1e-3, 1e-2, 1e-2, 1e-2, 1e-2, 1e-3, 1e-2],
  'ent_coef':     0.001,
}

PPO_Config_mnet_check_ss = {
  'vf_coef': 0.5,
  'max_grad_norm': 0.5,
  'distill_coef': [10.0, 10.0, 1e-4, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
  'ent_coef':     [0.0,  0.0,  1e-3,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
}

PPO_Config_mnet_check_ms = {
  'vf_coef': 0.5,
  'max_grad_norm': 0.5,
  'distill_coef': [10.0, 10.0, 10.0, 1e-4, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
  'ent_coef':     [0.0,  0.0,  0.0,  1e-3,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
}

PPO_Config_mnet_check_sft = {
  'vf_coef': 0.5,
  'max_grad_norm': 0.5,
  'distill_coef': [10.0, 10.0, 10.0, 10.0, 1e-4, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
  'ent_coef':     [0.0,  0.0,  0.0,  0.0,  1e-3,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
}

PPO_Config_mnet_check_cmd_u = {
  'vf_coef': 0.5,
  'max_grad_norm': 0.5,
  'distill_coef': [10.0, 10.0, 10.0, 10.0, 10.0, 1e-4, 10.0, 10.0, 10.0, 10.0, 10.0],
  'ent_coef':     [0.0,  0.0,  0.0,  0.0,  0.0,  1e-3,  0.0,  0.0,  0.0,  0.0,  0.0],
}

PPO_Config_mnet_check_pos = {
  'vf_coef': 0.5,
  'max_grad_norm': 0.5,
  'distill_coef': [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1e-4, 10.0, 10.0, 10.0, 10.0],
  'ent_coef':     [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1e-3,  0.0,  0.0,  0.0,  0.0],
}

PPO_Config_mnet_check_creep_pos = {
  'vf_coef': 0.5,
  'max_grad_norm': 0.5,
  'distill_coef': [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1e-4, 10.0, 10.0, 10.0],
  'ent_coef':     [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1e-3,  0.0,  0.0,  0.0],
}

PPO_Config_mnet_check_nydus_pos = {
  'vf_coef': 0.5,
  'max_grad_norm': 0.5,
  'distill_coef': [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1e-4, 10.0, 10.0],
  'ent_coef':     [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1e-3,  0.0,  0.0],
}

PPO_Config_mnet_check_base = {
  'vf_coef': 0.5,
  'max_grad_norm': 0.5,
  'distill_coef': [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1e-4, 10.0],
  'ent_coef':     [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1e-3,  0.0],
}

PPO_Config_mnet_check_unload = {
  'vf_coef': 0.5,
  'max_grad_norm': 0.5,
  'distill_coef': [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1e-4],
  'ent_coef':     [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1e-3],
}
