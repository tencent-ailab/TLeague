# Examples that Run in a Single Machine
We provide examples for how to run CSP-MARL training in a single machine for a couple of benchmark environments.

## Pong-2p
<img src="./pic-pong2p.png" alt="pong-2p" width="222" height="159">

Pong-2p is a simple environment that is good for sanity check.
It is a two-agent competitive game for pong.
For each agent,
the observation is a (84, 84, 4) stacked image of screen pixels,
and the action is Discrete(6).
See Appendix I of https://arxiv.org/abs/1907.09467

See the following examples for training with Pong-2p:
* [SelfPlay+PPO](EXAMPLE_PONG2P_SP_PPO.md)

## StarCraft II
<img src="./pic-sc2.png" alt="sc2" width="222" height="125">

See the following:
* [SelfPlay+PPO2](EXAMPLE_SC2_SP_PPO2.md)
* [SelfPlay+PPO+InfServer](EXAMPLE_SC2_SP_PPO_INFSERVER.md)
* [SelfPlay+Vtrace](EXAMPLE_SC2_SP_VTRACE.md)

See the following examples for Imitation Learning:
* [IL](EXAMPLE_SC2_IL.md)
* [IL+InfServer](EXAMPLE_SC2_IL_INFSERVER.md)

## ViZDoom
<img src="./pic-vizdoom.png" alt="vd" width="222" height="161">

See the following:
* [SelfPlay+PPO](EXAMPLE_VD_SP_PPO.md)

## Pommerman
<img src="./pic-pommerman.png" alt="pom" width="222" height="161">

See the following:
* [PFSP+PPO](EXAMPLE_POM_PFSP_PPO.md)

## Terminology
Through all the examples above, we use the terminology:
* `unroll_length`: how long the trajectory is when computing the RL Value Function using bootstrap. It must be a multiple of `batch_size`.
* `rollout_length`: the length for RNN BPTT. `rollout_length`= `rollout_len` in the `policy_config`.
* `rm_size`: size for Replay Memory. It must be a multiple of `unroll_length`.