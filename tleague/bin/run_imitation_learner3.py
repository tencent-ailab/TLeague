try:
  import horovod.tensorflow as hvd
  has_hvd = True
except:
  has_hvd = False
from absl import app
from absl import flags

from tleague.learners.imitation_learner3 import ImitationLearner3
from tleague.utils import read_config_dict
from tleague.utils import import_module_or_data


FLAGS = flags.FLAGS
flags.DEFINE_multi_string(
  "learner_spec", [
    "0:10001:10002,1:10003:10004",
    "0:10001:10002,2:10003:10004,3:10005:10006",
    ],
  "Comma separated gpu_id:port1:port2 triplet, which can occur as many times as"
  " the number of GPUs used in a host. This arg can occur as many times as the"
  " number of remote machines (in different IPs). The gpu id can be -1.",
  short_name='l'
)
flags.DEFINE_string("model_pool_addrs", "localhost:10003:10004",
                    "Model Pool address.")
flags.DEFINE_string("replay_filelist", None,
                    "Replay filelist for training and validation, csv format.")
flags.DEFINE_string("checkpoints_dir", './checkpoints',
                    "Checkpoints directory.")
flags.DEFINE_string("restore_checkpoint_path", None,
                    "Checkpoint path to restore.")
flags.DEFINE_float("learning_rate", 1e-4, "ADAM learning rate.")
flags.DEFINE_float("max_clip_grad_norm", 0, "max_grad_norm for clip variables")
flags.DEFINE_integer("min_train_sample_num", 10000,
                     "Minimal number of samples to start training.")
flags.DEFINE_integer("min_val_sample_num", 4000,
                     "Minimal number of samples to start validation.")
flags.DEFINE_integer("batch_size", 8, "Batch size per GPU.")
flags.DEFINE_integer("unroll_length", 32, "unroll length")
flags.DEFINE_integer("rollout_length", 1, "sample n frames consecutively.")
flags.DEFINE_integer("print_interval", 500, "Interval to print train loss.")
flags.DEFINE_integer("checkpoint_interval", 50000,
                     "Interval to save checkpoint.")
flags.DEFINE_integer("num_val_batches", 500, "Number of batches to validate.")
flags.DEFINE_integer("rm_size", 64000*32, "Replay memory size.")
flags.DEFINE_integer("train_generator_worker_num", 8,
                     "train generator worker num.")
flags.DEFINE_integer("val_generator_worker_num", 2,
                     "validation generator worker num.")
flags.DEFINE_boolean("enable_validation", True, "Should enable validation.")
flags.DEFINE_boolean("repeat_training_task", False,
                     "Whether repeatedly sending training task.")
flags.DEFINE_string("post_process_data", None,
                    "post process of (X, A), drop useless mask in SC2.")
flags.DEFINE_string("replay_converter",
                    "timitate.lib5.pb2all_converter.PB2AllConverter",
                    "replay converter used.")
flags.DEFINE_string("converter_config", "{}", "config used for converter")
flags.DEFINE_string("policy", "tpolicies.convnet.pure_conv.PureConv",
                    "policy used")
flags.DEFINE_string("policy_config", "", "config used for policy")
flags.DEFINE_string("after_loading_init_scope", None,
                    "tf scope for the variables that are re-initialized AFTER "
                    "loading model parameters from hard-disk. Can be used to "
                    "continue training from partially loaded models. "
                    "Regexpr supported as it uses tf.get_collection to get "
                    "variable list for re-initialization. For example, "
                    ".*/heads/.*"
                    "means re-init only those variables of the "
                    "action head layers (suppose they are under the heads/ name"
                    " scope)")
flags.DEFINE_integer("pub_interval", 500,
                     "freq of pub model to actors, in num of batch")
flags.DEFINE_boolean("use_mixed_precision", False,
                     "Whether to use mixed precision.")
flags.DEFINE_boolean("use_sparse_as_dense", True,
                     "Whether to use sparse_as_dense in hvd optimizer.")
flags.mark_flag_as_required("replay_filelist")


def main(_):
  if has_hvd:
    hvd.init()

  all_learner_specs = []
  for item in FLAGS.learner_spec:
    all_learner_specs += item.split(',')
  this_learner_ind = 0 if not has_hvd else hvd.rank()
  local_learner_spec = all_learner_specs[this_learner_ind]
  gpu_id_ports = local_learner_spec.split(':')  # gpu_id:port1:port2
  gpu_id, learner_ports = int(gpu_id_ports[0]), gpu_id_ports[1:]

  replay_converter_type = import_module_or_data(FLAGS.replay_converter)
  converter_config = read_config_dict(FLAGS.converter_config)
  policy = import_module_or_data(FLAGS.policy)
  policy_config = read_config_dict(FLAGS.policy_config)

  post_process_data = None
  if FLAGS.post_process_data is not None:
    post_process_data = import_module_or_data(FLAGS.post_process_data)

  learner = ImitationLearner3(
    ports=learner_ports,
    gpu_id=gpu_id,
    policy=policy.net_build_fun,
    policy_config=policy_config,
    policy_config_type=policy.net_config_cls,
    replay_filelist=FLAGS.replay_filelist,
    batch_size=FLAGS.batch_size,
    min_train_sample_num=FLAGS.min_train_sample_num,
    min_val_sample_num=FLAGS.min_val_sample_num,
    rm_size=FLAGS.rm_size,
    learning_rate=FLAGS.learning_rate,
    print_interval=FLAGS.print_interval,
    replay_converter_type=replay_converter_type,
    converter_config=converter_config,
    checkpoint_interval=FLAGS.checkpoint_interval,
    num_val_batches=FLAGS.num_val_batches,
    checkpoints_dir=FLAGS.checkpoints_dir,
    restore_checkpoint_path=FLAGS.restore_checkpoint_path,
    train_generator_worker_num=FLAGS.train_generator_worker_num,
    val_generator_worker_num=FLAGS.val_generator_worker_num,
    repeat_training_task=FLAGS.repeat_training_task,
    unroll_length=FLAGS.unroll_length,
    rollout_length=FLAGS.rollout_length,
    model_pool_addrs=FLAGS.model_pool_addrs.split(','),
    pub_interval=FLAGS.pub_interval,
    after_loading_init_scope=FLAGS.after_loading_init_scope,
    max_clip_grad_norm=FLAGS.max_clip_grad_norm,
    use_mixed_precision=FLAGS.use_mixed_precision,
    use_sparse_as_dense=FLAGS.use_sparse_as_dense,
    enable_validation=FLAGS.enable_validation,
    post_process_data=post_process_data
  )
  learner.run()


if __name__ == '__main__':
  app.run(main)
