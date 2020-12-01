from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import importlib

from absl import app
from absl import flags

from tleague.utils import read_config_dict
from tleague.utils import import_module_or_data
from tleague.utils import kill_sc2_processes_v2
from tleague.actors.replay_actor import ReplayActor

FLAGS = flags.FLAGS
flags.DEFINE_string("learner_addr", "localhost:10001:10002",
                    "Learner address.")
flags.DEFINE_string("model_pool_addrs", "localhost:10003:10004",
                    "Model Pool address.")
# replay & game stuff
flags.DEFINE_string("replay_converter",
                    "timitate.lib.pb2all_converter.PB2AllConverter",
                    "replay converter used.")
flags.DEFINE_string("converter_config", "{}", "config used for converter")
flags.DEFINE_string("replay_dir", None, "Root dir of replay files.")
flags.DEFINE_string("SC2_bin_root", '/root', "Root path for mv game core files.")
flags.DEFINE_string("game_version", '4.7.1', "Game core version.")
# training stuff
flags.DEFINE_string("policy", None, "policy used")
flags.DEFINE_string("policy_config", "", "config used for policy")
flags.DEFINE_string("agent", "tleague.actors.agent.PGAgent", "agent used.")
flags.DEFINE_string("infserver_addr", "", "infserver_addr agent used.")
flags.DEFINE_string("post_process_data", None,
                    "post process of (X, A), drop useless mask in SC2.")
flags.DEFINE_boolean("compress", True, "whether data is compressed for infserver")
flags.DEFINE_integer("update_model_freq", 32, "update model every n steps")
flags.DEFINE_integer("n_v", 1, "value length")
flags.DEFINE_integer("unroll_length", 32, "Push data to learner every n samples")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_float('data_augment_rate', -1.,
                   "Process data augmentation according to a ratio")
flags.DEFINE_integer('unk_mmr_dft_to', 4000, "Unknown MMR defaults to")
# logging
flags.DEFINE_integer("log_interval", 50, "Frequency of log in steps.")
flags.DEFINE_integer("log_interval_steps", 51,
                     "frequency of printing log in steps")
flags.DEFINE_boolean(
  "reboot_on_failure", False,
  "Should actor reboot on failure. NOTE: Before rebooting, it kills ALL the "
  "children SC2 processes on the machine. Use it carefully. Some hints:"
  "For k8s, we don't need rebooting as k8s can do it. "
)
flags.mark_flag_as_required("replay_dir")
flags.mark_flag_as_required("policy_config")


def main(_):
  converter_module, converter_name = FLAGS.replay_converter.rsplit(".", 1)
  replay_converter_type = getattr(importlib.import_module(converter_module),
                                  converter_name)
  converter_config = read_config_dict(FLAGS.converter_config)
  policy = None
  policy_config = None
  model_pool_addrs = None
  agent = None
  post_process_data = None
  if FLAGS.post_process_data is not None:
    post_process_data = import_module_or_data(FLAGS.post_process_data)
  if FLAGS.policy:
    policy = import_module_or_data(FLAGS.policy)
    policy_config = read_config_dict(FLAGS.policy_config)
    assert FLAGS.model_pool_addrs is not None
    model_pool_addrs = FLAGS.model_pool_addrs.split(',')
    agent = import_module_or_data(FLAGS.agent)
  actor = ReplayActor(learner_addr=FLAGS.learner_addr,
                      replay_dir=FLAGS.replay_dir,
                      replay_converter_type=replay_converter_type,
                      log_interval=FLAGS.log_interval,
                      step_mul=FLAGS.step_mul,
                      n_v=FLAGS.n_v,
                      game_version=FLAGS.game_version,
                      unroll_length=FLAGS.unroll_length,
                      policy=policy,
                      policy_config=policy_config,
                      model_pool_addrs=model_pool_addrs,
                      update_model_freq=FLAGS.update_model_freq,
                      converter_config=converter_config,
                      SC2_bin_root=FLAGS.SC2_bin_root,
                      agent_cls=agent,
                      infserver_addr=FLAGS.infserver_addr or None,
                      compress=FLAGS.compress,
                      post_process_data=post_process_data,
                      da_rate=FLAGS.data_augment_rate,
                      unk_mmr_dft_to=FLAGS.unk_mmr_dft_to)

  n_failures = 0
  while True:
    try:
      actor.run()
    except Exception as e:
      if not FLAGS.reboot_on_failure:
        raise e
      print("Actor crushed no. {}, the exception:\n{}".format(n_failures, e))
      n_failures += 1
      print("Rebooting...")
      kill_sc2_processes_v2()


if __name__ == '__main__':
  app.run(main)
