from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from tleague.league_mgrs.league_mgr import LeagueMgr, PARLeagueMgr
from tleague.utils import read_config_dict

FLAGS = flags.FLAGS
flags.DEFINE_string("port", "10005", "League manager port.")
flags.DEFINE_string("model_pool_addrs", "localhost:10003:10004",
                    "Model Pool address.")
flags.DEFINE_string("mutable_hyperparam_type",
                    "MutableHyperparamRandPredefSC2V1",
                    "Mutable hyperparam class name. checkout "
                    "tleague.hyperparam_mgr.hyperparam_types for supported"
                    " classes.")
flags.DEFINE_string('hyperparam_config_name', "",
                    "Config used for Mutable hyperparam. "
                    "pass those in tleague.hyperparam_mgr.configs or "
                    "py expression")
flags.DEFINE_string("game_mgr_type", "tleague.game_mgr.game_mgrs.RandomGameMgr",
                    "game_mgr class name. checkout tleague.game_mgr.game_mgrs "
                    "for supported classes.")
flags.DEFINE_string("game_mgr_config", "", "game_mgr config.")
flags.DEFINE_string("restore_checkpoint_dir", "",
                    "checkpoint dir from which the tleague restores. ")
flags.DEFINE_string("save_checkpoint_root", "",
                    "checkpoint root dir to which the tleague saves. "
                    "Will create sub folders during run.")
flags.DEFINE_string("init_model_paths", None,
                    "initial model paths must be a list of pairs of model key and path")
flags.DEFINE_integer("save_interval_secs", 4800,
                     "checkpoint saving frequency in seconds.")
flags.DEFINE_boolean('mute_actor_msg', False,
                     'Whether print req/rep of actors')
flags.DEFINE_boolean('save_learner_meta', False,
                     "Whether save learner's optimizer's params")
# printing/logging
flags.DEFINE_integer("verbose", 21,
                     "verbosity. The smaller, the noisier. Reference:"
                     "10:DEBUG, 20:INFO, 30: WARN, 40: ERROR, 50:DISABLED")
# pseudo parallel
flags.DEFINE_integer("pseudo_learner_num", 0,
                     "pseudo parallel run pseudo_learner_num learners "
                     "with fewer real learners")


def main(_):
  game_mgr_config = read_config_dict(FLAGS.game_mgr_config)
  if FLAGS.init_model_paths is not None:
    init_model_paths = eval(FLAGS.init_model_paths)
  else:
    init_model_paths = []
  if FLAGS.pseudo_learner_num > 0:
    league_mgr = PARLeagueMgr(
      port=FLAGS.port,
      model_pool_addrs=FLAGS.model_pool_addrs.split(','),
      mutable_hyperparam_type=FLAGS.mutable_hyperparam_type,
      pseudo_learner_num=FLAGS.pseudo_learner_num,
      hyperparam_config_name=FLAGS.hyperparam_config_name or None,
      restore_checkpoint_dir=FLAGS.restore_checkpoint_dir or None,
      save_checkpoint_root=FLAGS.save_checkpoint_root or None,
      save_interval_secs=FLAGS.save_interval_secs,
      mute_actor_msg=FLAGS.mute_actor_msg,
      game_mgr_type=FLAGS.game_mgr_type,
      game_mgr_config=game_mgr_config,
      verbose=FLAGS.verbose,
      init_model_paths=init_model_paths,
      save_learner_meta=FLAGS.save_learner_meta,
    )
  else:
    league_mgr = LeagueMgr(
      port=FLAGS.port,
      model_pool_addrs=FLAGS.model_pool_addrs.split(','),
      mutable_hyperparam_type=FLAGS.mutable_hyperparam_type,
      hyperparam_config_name=FLAGS.hyperparam_config_name or None,
      restore_checkpoint_dir=FLAGS.restore_checkpoint_dir or None,
      save_checkpoint_root=FLAGS.save_checkpoint_root or None,
      save_interval_secs=FLAGS.save_interval_secs,
      mute_actor_msg=FLAGS.mute_actor_msg,
      game_mgr_type=FLAGS.game_mgr_type,
      game_mgr_config=game_mgr_config,
      verbose=FLAGS.verbose,
      init_model_paths=init_model_paths,
      save_learner_meta=FLAGS.save_learner_meta,
    )
  league_mgr.run()


if __name__ == '__main__':
  app.run(main)
