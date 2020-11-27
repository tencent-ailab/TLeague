""" rsync local folder to remote machines sepcified by the clust_spec CSV.  """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from multiprocessing import Pool
import multiprocessing
import pexpect
from functools import partial

from absl import app
from absl import flags

from . import parse_workers_spec_csv
from . import is_remote_ip


FLAGS = flags.FLAGS
flags.DEFINE_integer('n_process', 1, 'number of parallel processes.')
flags.DEFINE_integer('timeout', 10, 'number of parallel processes.')
flags.DEFINE_string('clust_spec_csv_path', '../sandbox/clust_spec_example.csv',
                    'cluster specification csv file path.')
flags.DEFINE_string('source', '/home/work/TLeague',
                    'local source')
flags.DEFINE_string('target', '/home/work',
                    'remote target')


def _func_run_cmd(q_failed_workers, w):
 try:
  if is_remote_ip(w.ip):
    command = 'rsync -az -e "ssh -p {}" {} {}@{}:{}'.format(
      w.ssh_port, FLAGS.source, w.ssh_username, w.ip, FLAGS.target)
    print(command)
    child = pexpect.spawn(command, timeout=FLAGS.timeout)
    ret = child.expect([pexpect.TIMEOUT, 'Are you sure you want to continue connecting', '[P|p]assword'])
    if ret == 0:
      print('[-] Error Connecting')
      q_failed_workers.put((w, 'Timeout'))
      return
    elif ret == 1:
      child.sendline('yes')
      ret = child.expect([pexpect.TIMEOUT, '[P|p]assword'])
      if ret == 0:
        print('[-] Error Connecting')
        q_failed_workers.put((w, 'Timeout'))
        return
    child.sendline(w.ssh_password)
    ret = child.expect([pexpect.EOF, 'Permission denied'])
    if ret == 0:
      return
    else:
      print('Permission denied')
      q_failed_workers.put((w, 'Permission denied'))
  else:
    print('{} is local machine, skipping...', w)
    return
 except:
  print(w)

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

