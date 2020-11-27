""" execute command in remote machines sepcified by the clust_spec CSV.  """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from multiprocessing import Pool
import multiprocessing
from functools import partial

from absl import app
from absl import flags

from . import parse_workers_spec_csv
from . import is_remote_ip
from . import run_cmd_remote


FLAGS = flags.FLAGS
flags.DEFINE_integer('n_process', 1, 'number of parallel processes.')
flags.DEFINE_string('clust_spec_csv_path', '../sandbox/clust_spec_example.csv',
                    'cluster specification csv file path.')
flags.DEFINE_multi_string(
  "remote_cmd", ["pwd", "\"ls -lart\"", "\"echo \\\"hello world\\\" \""],
  "command to be executed in remote machine. Use backslash \\ and "
  "quote \" carefully. It can occur multiple times. E.g., "
  "-r pwd\n"
  "-r \"ls -lart\"\n"
  "-r \"echo \\\"hello world\\\"\"\n"
  "...",
  short_name='r'
)


def _func_run_cmd(q_failed_workers, w):
  try:
    if is_remote_ip(w.ip):
      for cmd in FLAGS.remote_cmd:
        run_cmd_remote(cmd=cmd, ip=w.ip, port=w.ssh_port,
                       username=w.ssh_username, password=w.ssh_password)
    else:
      print('{} is local machine, skipping...', w)
  except Exception as e:
    q_failed_workers.put((w, e))


def main(_):
  workers = parse_workers_spec_csv(FLAGS.clust_spec_csv_path)

  q_failed_workers = multiprocessing.Manager().Queue()
  mp = Pool(processes=FLAGS.n_process)
  tmp_func = partial(_func_run_cmd, q_failed_workers)
  mp.map(tmp_func, workers)
  mp.close()
  mp.join()

  print('\nall commands sent. Succeeded: {}, Failed: {}'.format(
    len(workers) - q_failed_workers.qsize(), q_failed_workers.qsize()
  ))
  for _ in range(q_failed_workers.qsize()):
    w, e = q_failed_workers.get()
    print(w)
    print('{}: {}'.format(type(e), e))


if __name__ == '__main__':
  app.run(main)

