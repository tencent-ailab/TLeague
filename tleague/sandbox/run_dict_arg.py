import importlib
from time import sleep

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("policy_config", "{}", "test config")

def main(_):
  c = eval(FLAGS.policy_config)
  print(c, flush=True)
  sleep(360000)


if __name__ == '__main__':
  app.run(main)
