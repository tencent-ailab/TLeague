from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
from absl import app
from absl import flags

from tleague.league_mgrs.league_mgr_apis import LeagueMgrAPIs
from tleague.model_pools.model import Model


FLAGS = flags.FLAGS
flags.DEFINE_string("league_mgr_addr", "localhost:10005",
                    "League manager address.")

flags.DEFINE_string("model_path", "", "model file path")
flags.DEFINE_string("model_key", None, "rename model.key to model_key if not None")


def main(_):
  league_mgr_apis = LeagueMgrAPIs(FLAGS.league_mgr_addr)
  with open(FLAGS.model_path, 'rb') as f:
    model = pickle.load(f)
  if not isinstance(model, Model):
    model = Model(model, None, None)
  if not model.is_freezed():
    model.freeze()
  if FLAGS.model_key:
    model.key = FLAGS.model_key
  league_mgr_apis.request_add_model(model)


if __name__ == '__main__':
  app.run(main)
