""" kill processes on the ports and tmux session on each (local or remote)
 machine specified in the cluster spec CSV file.  """
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
from . import to_cmd_str
from . import run_cmd_local
from . import run_cmd_remote


FLAGS = flags.FLAGS
flags.DEFINE_integer('n_process', 1, 'number of parallel processes.')
flags.DEFINE_string('clust_spec_csv_path', '../sandbox/clust_spec_example.csv',
                    'cluster specification csv file path.')
flags.DEFINE_string('tmux_sess', 'tlea', 'tmux session name for each machine')


def create_port_cmd(worker):
  cmds = []
  # kill processes on ports
  if worker.port1:
    #cmds += ["kill $( lsof -i:{} -t ) > /dev/null 2>&1".format(worker.port1),
    #         "&&"]
    cmds += ["kill $( lsof -i:{} -t )".format(worker.port1)]
  if worker.port2:
    cmds += ["&&", "kill $( lsof -i:{} -t )".format(worker.port2)]
  return to_cmd_str(cmds)


def create_tmux_cmd(worker):
  cmds = []
  # kill tmux session
  cmds += ["tmux kill-session -t {}".format(FLAGS.tmux_sess)]
  return to_cmd_str(cmds)


def main_single_old(_):
  workers = parse_workers_spec_csv(FLAGS.clust_spec_csv_path)

  failed_workers = []
  for w in workers:
    port_cmd = create_port_cmd(w)
    tmux_cmd = create_tmux_cmd(w)
    try:
      if is_remote_ip(w.ip):
        run_cmd_remote(cmd=port_cmd, ip=w.ip, port=w.ssh_port,
                       username=w.ssh_username, password=w.ssh_password)
        run_cmd_remote(cmd=tmux_cmd, ip=w.ip, port=w.ssh_port,
                       username=w.ssh_username, password=w.ssh_password)
      else:
        run_cmd_local(cmd=port_cmd)
        run_cmd_local(cmd=tmux_cmd)
    except Exception:
      failed_workers.append(w)

  print('\nall killing commands sent. Succeeded: {}, Failed: {}'.format(
    len(workers) - len(failed_workers), len(failed_workers)
  ))
  print(failed_workers)


def _func_run_cmd(q_failed_workers, w):
  port_cmd = create_port_cmd(w)
  tmux_cmd = create_tmux_cmd(w)
  try:
    if is_remote_ip(w.ip):
      run_cmd_remote(cmd=port_cmd, ip=w.ip, port=w.ssh_port,
                     username=w.ssh_username, password=w.ssh_password)
      run_cmd_remote(cmd=tmux_cmd, ip=w.ip, port=w.ssh_port,
                     username=w.ssh_username, password=w.ssh_password)
    else:
      run_cmd_local(cmd=port_cmd)
      run_cmd_local(cmd=tmux_cmd)
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

  n_sucess = len(workers) - q_failed_workers.qsize()
  n_failure = q_failed_workers.qsize()
  for _ in range(q_failed_workers.qsize()):
    w, e = q_failed_workers.get()
    print(w)
    print('{}: {}'.format(type(e), e))
  print('\nall commands sent. Succeeded: {}, Failed: {}'.format(
    n_sucess, n_failure
  ))



if __name__ == '__main__':
  app.run(main)

