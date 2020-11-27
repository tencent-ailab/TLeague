from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

from absl import app
from absl import flags

from tleague.model_pools.model_pool import ModelPool


FLAGS = flags.FLAGS
flags.DEFINE_string("ports", "10003:10004", "Ports.")
flags.DEFINE_integer("verbose", 50,
                     "verbosity. The smaller, the noisier. Reference:"
                     "10:DEBUG, 20:INFO, 30: WARN, 40: ERROR, 50:DISABLED")


def main(_):
  model_pool = ModelPool(ports=FLAGS.ports, verbose=FLAGS.verbose)
  model_pool.run()


if __name__ == '__main__':
  app.run(main)
