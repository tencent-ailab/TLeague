from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
from absl import app
from absl import flags

from tleague.model_pools.model_pool_apis import ModelPoolAPIs
from tleague.model_pools.model import Model


FLAGS = flags.FLAGS
flags.DEFINE_string("model_pool_addrs", "localhost:10003:10004",
                    "Model Pool address.")

flags.DEFINE_multi_string("model_path", [], "model file path")
flags.DEFINE_multi_string("model_key", [], "model_keys")


def main(_):
  model_pool_apis = ModelPoolAPIs(FLAGS.model_pool_addrs.split(','))
  keys = model_pool_apis.pull_keys()
  for key, model_path in zip(FLAGS.model_key, FLAGS.model_path):
    if key in keys:
      m = model_pool_apis.pull_model(key)
      with open(model_path, 'rb') as f:
        model = pickle.load(f)
      if isinstance(model, Model):
        model = model.model
      model_pool_apis.push_model(model, m.hyperparam, m.key, m.createtime,
                                 m.freezetime, m.updatetime)


if __name__ == '__main__':
  app.run(main)
