#!/usr/bin/env python
""" Hyperparam configs for different env and experiments.

Suggested Convention:
WhatTheHyperparamType_Config_SuffixForGameAndOrVersion """


MutableHyperparamRandPredefSC2V2_Config_v1 = {
  'learning_rate': 1e-4,
  'reward_scale': 0.005,
  'available_sigma': (1/0.6, 1/6.0, 1/60.0,),
  'initial_sigma': 1/6.0,
}

MutableHyperparamRandPerturb_Config_SC2 = {
  'learning_rate': 1e-4,
  'reward_scale': 0.05,
}

MutableHyperparamRandPerturb_Config_SC2v2 = {
  'learning_rate': 1e-4,
  'reward_scale': 0.05,
  'reward_len': 11,
  'available_sigma': (1/0.6, 1/6.0, 1/60.0,),
  'initial_sigma': 1/6.0,
}

MutableHyperparamPartialPerturb_Config_SC2v1 = {
  'learning_rate': 1e-5,
  'lam': 0.99,
  'gamma': 0.999,
  'reward_scale': 0.1,
  'available_sigma': (1/0.6, 1/6.0, 1/60.0,),
  'initial_sigma': 1/6.0,
  'initial_reward_weights': [[1] + [0] * 13 + [0, 0, 0],
                             [1] + [0] * 13 + [0, 0.001, 0],
                             [1] + [0] * 13 + [0, 0, 0.001]],
  'perturb_ind': tuple([(0, i) for i in range(1, 14)])
}

MutableHyperparamPartialPerturb_Config_SC2v2 = {
  'learning_rate': 1e-5,
  'lam': 0.99,
  'gamma': 0.999,
  'reward_scale': 0.008,
  'available_sigma': (1/0.6, 1/6.0, 1/60.0,),
  'initial_sigma': 1/6.0,
  'initial_reward_weights': [[1] + [0] * 13 + [0, 0, 0],
                             [1] + [0] * 13 + [0, 0.001, 0],
                             [1] + [0] * 13 + [0, 0, 0.001]],
  'perturb_ind': tuple([(0, i) for i in range(1, 14)])
}

MutableHyperparamPreDefPartialPerturb_Config_SC2v1 = {
  'learning_rate': 1e-5,
  'lam': 0.99,
  'gamma': 0.999,
  'reward_scale': 0.05,  # multiplied on the entry in perturb_ind
  'available_sigma': (1/0.6, 1/6.0, 1/60.0,),
  'initial_sigma': 1/6.0,
  'initial_reward_weights': # 5 pre-difined initial_reward_weight
    [[1] + [1.0, 0.5, 1.0, 1.0, 0.2, 0.1, 1.0, 0.8, 1.0, 0.5, 1.0, 0.2, 1.0] + [0.0, 0.0, 0.001],
     [1] + [2.0, 1.0, 1.0, 1.0, 0.5, 0.1, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1] + [0.0, 0.0, 0.001],
     [1] + [1.0, 0.5, 1.0, 2.0, 1.0, 0.3, 2.0, 1.0, 0.2, 0.2, 0.1, 0.1, 2.0] + [0.0, 0.0, 0.001],
     [1] + [1.0, 0.1, 1.0, 0.5, 0.1, 0.1, 0.1, 2.0, 0.5, 1.0, 0.1, 0.5, 0.1] + [0.0, 0.0, 0.001],
     [1] + [1.0, 0.1, 1.0, 1.0, 0.2, 0.1, 0.5, 0.5, 2.0, 1.0, 2.0, 1.0, 0.5] + [0.0, 0.0, 0.001],],
  'perturb_ind': range(1, 14),
  'distill_model_key': 'rand_model:0001'
}

MutableHyperparamPreDefPartialPerturb_Config_SC2v2 = {
  'learning_rate': 1e-5,
  'lam': 0.99,
  'gamma': 0.999,
  'reward_scale': 0.01,  # multiplied on the entry in perturb_ind
  'available_sigma': (1/0.6, 1/6.0, 1/60.0,),
  'initial_sigma': 1/6.0,
  'initial_reward_weights': # 5 pre-difined initial_reward_weight
    [[1.0] + [0.2, 0.2, 1.0, 1.0, 0.2, 0.1, 1.0, 0.8, 1.0, 0.5, 1.0, 0.2, 1.0] + [0.0, 0.0, 0.005],
     [1.0] + [2.0, 1.0, 1.0, 1.0, 0.5, 0.1, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1] + [0.0, 0.0, 0.005],
     [1.0] + [0.0, 0.0, 1.0, 2.0, 1.0, 0.3, 2.0, 1.0, 0.2, 0.2, 0.1, 0.1, 2.0] + [0.0, 0.0, 0.005],
     [1.0] + [0.0, 0.0, 1.0, 1.0, 0.2, 0.1, 0.5, 0.5, 2.0, 1.0, 2.0, 1.0, 0.5] + [0.0, 0.0, 0.005],
     [1.0] + [0.0, 0.0, 1.0, 1.0, 0.2, 0.1, 0.5, 0.5, 2.0, 1.0, 2.0, 1.0, 0.5] + [0.0, 0.0, 0.005],
     [1.0] + [0.0] * 13 + [0.0, 0.0, 0.0],
     [1.0] + [0.0] * 13 + [0.0, 0.0, 0.005],],
  'perturb_ind': range(1, 14),
  'distill_model_key': 'None:init_model'
}

MutableHyperparamPreDefPartialPerturb_Config_SC2_tr0925 = {
  'learning_rate': 0.00001,
  'lam': 0.99,
  'gamma': 0.99975,
  'reward_scale': 0.01,
  'available_sigma': [1/0.6, 1/6.0, 1/60.0],
  'initial_sigma': 1/6.0,
  'initial_reward_weights':
    [[1.0] + [-0.1, 0.2, 1.0, 1.0, 0.2, 0.1, 1.0, 0.8, 1.0, 0.5, 1.0, 0.2, 1.0] + [0.0, 0.0, 0.005],
     [1.0] + [-0.1, 1.0, 1.0, 1.0, 0.5, 0.1, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1] + [0.0, 0.0, 0.005],
     [1.0] + [0.0, 0.0, 1.0, 2.0, 1.0, 0.3, 2.0, 1.0, 0.2, 0.2, 0.1, 0.1, 2.0] + [0.0, 0.0, 0.005],
     [1.0] + [0.0, 0.0, 1.0, 1.0, 0.2, 0.1, 0.5, 0.5, 2.0, 1.0, 2.0, 1.0, 0.5] + [0.0, 0.0, 0.005],
     [1.0] + [0.0, 0.0, 1.0, 1.0, 0.2, 0.1, 0.5, 0.5, 2.0, 1.0, 2.0, 1.0, 0.5] + [0.0, 0.0, 0.005],
     [1.0] + [0.0] * 13 + [0.0, 0.0, 0.0],
     [1.0] + [0.0] * 13 + [0.0, 0.0, 0.005],],
  'perturb_ind': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
  'distill_model_key': 'None:init_model',
  'max_total_timesteps': 70000000,
  'default_burn_in_timesteps': 5400000,
  'minimal_lp_len_ratio': 0.3,
  'need_burn_in': 0,
}

MutableHyperparamRandPredefSigma_Config_SC2v1 = {
  'learning_rate': 1e-5,
  'lam': 0.99,
  'gamma': 0.999,
  'reward_weights': [[1, 0, 0, 0],
                     [1, 0, 0.001, 0],
                     [1, 0, 0, 0.001]],
  'available_sigma': (1/0.6, 1/6.0, 1/60.0,),
  'initial_sigma': 1/6.0,
}

ConstantHyperparam_Config_SC2 = {
  'learning_rate': 1e-5,
  'lam': 0.99,
  'gamma': 0.999,
  'reward_weights': [[1, 0, 0, 0],
                     [1, 0, 0.001, 0],
                     [1, 0, 0, 0.001]]
}

ConstantHyperparam_Config_SC2FullFormal = {
  'learning_rate': 1e-4,  # can be lambda expression
  'lam': 0.99,
  'gamma': 0.999,
  'sigma': 1,
  'reward_weights': [1, 0.001, 0.001, 0.005]
}

ConstantHyperparam_Config_SC2FullFormal_v2 = {
  'learning_rate': 1e-5,
  'lam': 0.99,
  'gamma': 0.999,
  'sigma': 10.0,
  'reward_weights': [1, 0.001, 0.001, 0.005]
}

ConstantHyperparam_Config_SC2FullFormal_v3 = {
  'learning_rate': 1e-6,
  'lam': 0.99,
  'gamma': 0.99,
  'sigma': 10.0,
  'reward_weights': [1, 0.001, 0.001, 0.005]
}

ConstantHyperparam_Config_SC2FullFormal_v4 = {
  'learning_rate': 1e-6,
  'lam': 0.99,
  'gamma': 0.999,
  'sigma': 10.0,
  'reward_weights': [1.0, 0.00, 0.00, 0.00]
}

ConstantHyperparam_Config_SC2FullFormal_v5 = {
  'learning_rate': 1e-5,
  'lam': 0.99,
  'gamma': 0.999,
  'sigma': 10.0,
  'reward_weights': [1.0, 0.00, 0.00, 0.00]
}

ConstantHyperparam_Config_SC2FullFormal_v6 = {
  'learning_rate': 1e-5,
  'lam': 0.99,
  'gamma': 0.99,
  'sigma': 10.0,
  'reward_weights': [1.0, 0.00, 0.00, 0.00]
}

ConstantHyperparam_Config_SC2FullFormal_v7 = {
  'learning_rate': 1e-6,
  'lam': 0.99,
  'gamma': 0.99,
  'sigma': 10.0,
  'reward_weights': [1, 0.001, 0.001, 0.005]
}

ConstantHyperparam_Config_SC2FullFormal_v8 = {
  'learning_rate': 1e-5,
  'lam': 0.99,
  'gamma': 0.999,
  'sigma': 10.0,
  'reward_weights': [1, 0.0, 0.0, 0.001]
}

ConstantHyperparam_Config_SC2FullFormal_v9 = {
  'learning_rate': 1e-5,
  'lam': 0.99,
  'gamma': 0.999,
  'sigma': 10.0,
  'reward_weights': [0] * 17
}

ConstantHyperparam_Config_SC2FullFormal_v10 = {
  'learning_rate': 1e-5,
  'lam': 0.99,
  'gamma': 0.999,
  'sigma': 10.0,
  'reward_weights': [1.0] + [a * 0.01 for a in [
    0.0, 0.0, 1.0, 1.0, 0.2, 0.1, 0.5, 0.5, 2.0, 1.0, 2.0, 1.0, 0.5]] + [0.0, 0.0, 0.005],
  'distill_model_key': 'None:init_model'
}

ConstantHyperparam_Config_SC2FullFormal_v11 = {
  'learning_rate': 1e-5,
  'lam': 0.99,
  'gamma': 0.999,
  'sigma': 10.0,
  'reward_weights': [1.0] + [0] * 13 + [0.0, 0.0, 0.005],
  'distill_model_key': 'None:init_model'
}

ConstantHyperparam_Config_SC2FullFormal_v12 = {
  'learning_rate': 1e-5,
  'lam': 0.99,
  'gamma': 0.999,
  'sigma': 10.0,
  'reward_weights': [1.0] + [0] * 13 + [0.0, 0.0, 0.0],
  'distill_model_key': 'None:init_model'
}

ConstantHyperparam_Config_SC2FullFormal_zerolr = {
  'learning_rate': 0.0,
  'lam': 0.99,
  'gamma': 0.999,
  'sigma': 10.0,
  'reward_weights': [1, 0.001, 0.001, 0.005]
}

ConstantHyperparam_Config_SC2FullFormal_v4_dict = {
  'learning_rate': 1e-5,
  'lam': 0.95,
  'gamma': 1,
  'sigma': 10,
  'reward_weights': [[1, 0, 0, 0, 0]],
  'distill_model_key': 'rand_model:0001',
}

ConstantHyperparam_Config_pommerman = {
  'learning_rate': 1e-5,  # can be lambda expression
  'lam': 0.95,
  'gamma': 0.99,
  'sigma': 1,
  'reward_weights': [1, 0.1, 0.1, 0, 0, 0.1, 0.1, 0.1, 0.1]
}

ConstantHyperparam_Config_soccer = {
  'learning_rate': 1e-5,  # can be lambda expression
  'lam': 0.95,
  'gamma': 0.99,
  'sigma': 1,
  'reward_weights': [1, 1, 0.0001, 0.0005, 1, 1, 0.0001, 0.0005]
}

ConstantHyperparam_Config_vizdoom = {
  'learning_rate': 1e-5,
  'lam': 0.95,
  'gamma': 0.99,
  'sigma': 10,
  'reward_weights': [1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.001]
}
