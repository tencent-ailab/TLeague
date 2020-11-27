import pickle
import time
import importlib
from absl import app
from absl import flags
from tleague.inference_server.server import InfServer
from tleague.inference_server.api import InfServerAPIs
from tleague.utils import import_module_or_data
from tleague.utils.data_structure import InfData, ILData
from tleague.utils.io import TensorZipper


FLAGS = flags.FLAGS
flags.DEFINE_string("role", 'Server', "run server or actor")
flags.DEFINE_integer("gpu_id", -1, "which gpu server use, -1 means no gpu")
flags.DEFINE_integer("port", 5678, "which port server use")
flags.DEFINE_integer("batch_size", 32, "batch_size server use")
flags.DEFINE_integer("pull_worker_num", 4, "pull_worker_num server use")
flags.DEFINE_string("server_addr", 'localhost:5678', "server hostname and port")
flags.DEFINE_bool("use_gpu_server", True, "whether actor use gpu inference server")


def main(_):
  policy = "tpolicies.net_zoo.mnet_v6.mnet_v6d6"
  policy_config = {
    'use_xla': True,
    'test': False,
    'use_loss_type': 'none',
    'use_value_head': False,
    'use_self_fed_heads': True,
    'use_lstm': True,
    'nlstm': 256,
    'hs_len': 256*2,
    'lstm_duration': 1,
    'lstm_dropout_rate': 0.0,
    'lstm_cell_type': 'lstm',
    'lstm_layer_norm': True,
    'weight_decay': 0.00002,
    'arg_scope_type': 'type_b',
    'endpoints_verbosity': 10,
    'n_v': 7,
    'distillation': True,
    'fix_all_embed': False,
    'use_base_mask': True,
    'zstat_embed_version': 'v3',
    'sync_statistics': 'horovod',
    'temperature': 0.8,
    'merge_pi': False,
  }
  converter_config = {
    'zstat_data_src': '/root/replay_ds/rp1522-mv-zstat',
    'input_map_size': (128, 128),
    'output_map_size': (128, 128),
    'delete_useless_selection': False,
    'dict_space': True,
    'max_bo_count': 50,
    'max_bobt_count': 20,
    'zstat_zeroing_prob': 0.1,
    'zmaker_version': 'v5',
  }
  policy = import_module_or_data(policy)
  replay_converter_name = "timitate.lib6.pb2all_converter.PB2AllConverter"
  converter_module, converter_name = replay_converter_name.rsplit(".", 1)
  replay_converter_type = getattr(importlib.import_module(converter_module),
                                  converter_name)
  replay_converter = replay_converter_type(**converter_config)
  ob_space, ac_space = replay_converter.space
  rnn = (False if 'use_lstm' not in policy_config
         else policy_config['use_lstm'])
  hs_len = (policy_config['hs_len'] if ('hs_len' in policy_config)
            else 2 * policy_config['nlstm'] if ('nlstm' in policy_config) else 128)
  ds = InfData(ob_space, ac_space, policy_config['use_self_fed_heads'], rnn, hs_len)
  cached_ds = ILData(ob_space, ac_space, rnn, hs_len)

  if FLAGS.role == 'Server':
    S = InfServer(None, None, FLAGS.port, ds, FLAGS.batch_size, ob_space, ac_space,
                  policy, policy_config=policy_config, gpu_id=FLAGS.gpu_id,
                  pull_worker_num=FLAGS.pull_worker_num)
    S.run()
  elif FLAGS.role == 'Actor':
    data = pickle.load(open('data', 'rb'))
    data_set = [cached_ds.make_structure(TensorZipper.decompress(d)) for d in data]
    data_set = [ds.structure(d.X, d.S, d.M) for d in data_set]
    n = len(data_set)
    policy_config['batch_size'] = 1
    policy_config['rollout_len'] = 1
    policy_config['use_loss_type'] = 'none'
    if FLAGS.use_gpu_server:
      from tleague.actors.agent import PGAgentGPU
      agent = PGAgentGPU(FLAGS.server_addr, ds, hs_len)
    else:
      from tleague.actors.agent import PGAgent2
      agent = PGAgent2(policy, ob_space, ac_space, policy_config=policy_config)
    while True:
      t0 = time.time()
      for sample in data_set:
        pred=agent.step(sample.X)
        # print(pred['A_AB'])
      cost = time.time()-t0
      print('Predict {} samples costs {} seconds, fps {}.'.format(n, cost, n/cost), flush=True)

if __name__ == '__main__':
  app.run(main)






