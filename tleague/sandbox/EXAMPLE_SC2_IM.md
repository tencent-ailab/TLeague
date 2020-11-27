# Examples for SC2 Imitation Learning

See `exampl_sc2_im.sh` for example of `sc2` imitation learning (without Inference Server).
See `exampl_sc2_im_infserver.sh` for example of `sc2` imitation learning with Inference Server.

Terminology:
* `unroll_length`: how long the trajectory is when computing the RL Value Function using bootstrap. It must be a multiple of `batch_size`.
* `rollout_length`: the length for RNN BPTT. `rollout_length`= `rollout_len` in the `policy_config`.
* `rm_size`: size for Replay Memory. It must be a multiple of `unroll_length`. .