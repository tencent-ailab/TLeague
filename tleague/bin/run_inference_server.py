from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
  import horovod.tensorflow as hvd
  has_hvd = True
except:
  has_hvd = False
from absl import app
from absl import flags

from tleague.utils import read_config_dict
from tleague.utils import import_module_or_data
from tleague.inference_server.server import InfServer
from tleague.utils.data_structure import InfData
from tleague.envs.create_envs import env_space


FLAGS = flags.FLAGS
flags.DEFINE_string("league_mgr_addr", "",
                    "League manager address.")
flags.DEFINE_string("model_pool_addrs", "localhost:10003:10004",
                    "Model Pool address.")
flags.DEFINE_string("learner_id", "",
                    "Learner id")

flags.DEFINE_integer("port", 5678, "port server used")
flags.DEFINE_integer("gpu_id", 0, "gpu server used, -1 means cpu only.")
flags.DEFINE_integer("log_interval", 100, "freq of print log, in num of batch")
flags.DEFINE_boolean("compress", True, "whether data is compressed")
flags.DEFINE_integer("batch_worker_num", 2, "batch_worker_num")
flags.DEFINE_boolean("is_rl", False, "whether in rl")
flags.DEFINE_string("env", "sc2", "task env")
flags.DEFINE_string("env_config", "",
                    "python dict config used for env. "
                    "e.g., {'replay_dir': '/root/replays/ext471_zvz'}")
flags.DEFINE_string("interface_config", "",
                    "python dict config used for Arena interface. "
                    "e.g., {'zstat_data_src_path': '/root/replay_ds/zstat'}")
flags.DEFINE_string("replay_converter",
                    "timitate.lib5.pb2all_converter.PB2AllConverter",
                    "replay converter used.")
flags.DEFINE_string("converter_config", "{}", "config used for converter")
flags.DEFINE_string("policy", "tpolicies.ppo.policies.DeepMultiHeadMlpPolicy",
                    "policy used")
flags.DEFINE_multi_string("policy_config", ["{}"], "config used for policy")
flags.DEFINE_multi_string("infserver_config", ["{}"], "config used for infserver")
flags.DEFINE_multi_string("post_process_data", [""],
                          "post process of (X, A), drop useless mask in SC2.")
flags.DEFINE_boolean("hvd_run", False, "whether to use horovod")


def main(_):
  if not has_hvd:
    print('horovod is unavailable, FLAGS.hvd_run will be ignored.')

  index = 0
  if has_hvd and FLAGS.hvd_run:
    hvd.init()
    index = hvd.local_rank()
    print('Horovod initialized.')
  port = FLAGS.port - index
  print('index: {}, using port: {}'.format(index, port), flush=True)

  policy = import_module_or_data(FLAGS.policy)
  policy_config = read_config_dict(
    FLAGS.policy_config[index % len(FLAGS.policy_config)]
  )
  infserver_config = read_config_dict(
    FLAGS.infserver_config[index % len(FLAGS.infserver_config)]
  )
  post_process_data = FLAGS.post_process_data[
                        index % len(FLAGS.post_process_data)] or None
  if FLAGS.is_rl:
    env_config = read_config_dict(FLAGS.env_config)
    interface_config = read_config_dict(FLAGS.interface_config)
    ob_space, ac_space = env_space(FLAGS.env, env_config, interface_config)
  else:
    replay_converter_type = import_module_or_data(FLAGS.replay_converter)
    converter_config = read_config_dict(FLAGS.converter_config)
    replay_converter = replay_converter_type(**converter_config)
    ob_space, ac_space = replay_converter.space.spaces
    if 'model_key' not in infserver_config:
      infserver_config['model_key'] = 'IL-model'
  if post_process_data is not None:
    post_process_data = import_module_or_data(post_process_data)
    ob_space, ac_space = post_process_data(ob_space, ac_space)
  nc = policy.net_config_cls(ob_space, ac_space, **policy_config)
  ds = InfData(ob_space, ac_space, nc.use_self_fed_heads, nc.use_lstm,
               nc.hs_len)

  server = InfServer(league_mgr_addr=FLAGS.league_mgr_addr or None,
                     model_pool_addrs=FLAGS.model_pool_addrs.split(','),
                     port=port,
                     ds=ds,
                     batch_size=policy_config['batch_size'],
                     ob_space=ob_space, ac_space=ac_space,
                     policy=policy,
                     policy_config=policy_config,
                     gpu_id=FLAGS.gpu_id,
                     compress=FLAGS.compress,
                     batch_worker_num=FLAGS.batch_worker_num,
                     learner_id=FLAGS.learner_id,
                     **infserver_config)
  server.run()


if __name__ == '__main__':
  app.run(main)
