from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple
import csv
import os
from os import path
from os.path import dirname
from time import sleep
import socket
import zipfile
import tempfile
import shlex

import paramiko

WorkerSpec = namedtuple(
  'WorkerSpec',
  'job ip port1 port2 cuda_visible_devices ssh_username ssh_password ssh_port, learner_id'
)
WorkerSpec.__new__.__defaults__ = ('',)  # default learner_id


def get_worker_addr(worker):
  if worker.port2:
    return ':'.join([worker.ip, worker.port1, worker.port2])
  return ':'.join([worker.ip, worker.port1])


def get_worker_ports(worker):
  if worker.port2:
    return ':'.join([worker.port1, worker.port2])
  return worker.port1


def pair_actors_learners(actors, learners):
  act_addr_to_lrn = {}
  # do it round-robin
  for i, actor in enumerate(actors):
    key = get_worker_addr(actor)
    val = learners[i % len(learners)]
    act_addr_to_lrn[key] = val  # NOTE: overwrite duplicate actor address
  return act_addr_to_lrn


def split_workers(workers):
  job_to_workers = {}
  for w in workers:
    key = w.job
    if job_to_workers.get(key, None) is None:
      job_to_workers[key] = []
    job_to_workers[key].append(w)
  return job_to_workers


def split_learners(learners):
  id_to_learners = {}
  for w in learners:
    key = w.learner_id
    if id_to_learners.get(key, None) is None:
      id_to_learners[key] = []
    id_to_learners[key].append(w)
  return id_to_learners


def parse_workers_spec_csv(csv_path):
  workers = []
  with open(csv_path, "r") as f:
    reader = csv.DictReader(f, delimiter=",")
    for i, line in enumerate(reader):
      workers.append(WorkerSpec(**line))
  return workers


def group_workers_by_ip(workers):
  # TODO(pengsun): fix for localhost
  ip_to_workers = {}
  for w in workers:
    if ip_to_workers.get(w.ip, None) is None:
      ip_to_workers[w.ip] = []
    ip_to_workers[w.ip].append(w)
  return ip_to_workers


def get_containing_folder():
  return dirname(dirname(dirname(os.path.abspath(__file__))))


def get_containing_folder_basename(cont_folder=''):
  if cont_folder:
    return os.path.basename(cont_folder)
  return os.path.basename(get_containing_folder())


def zip_containing_folder_relpath():
  """ recursively zip containing folder in relative path

  For example, "/Users/aaa/code/TLeague/*" will be zipped as "TLeague/*"
  """
  blacklist_file = ['.DS_Store']
  blacklist_ext = ['pyc']
  blacklist_dir = ['.git', '.idea', 'TLeague.egg-info', '__pycache__']
  whitelist_ext = ['.py', '.csv']

  def _should_skip(d, f=''):
    dd = os.path.basename(d)
    if dd in blacklist_dir:
      return True
    if f in blacklist_file:
      return True
    _, ext = os.path.splitext(f)
    if ext in blacklist_ext:
      return True
    return False

  def _should_zip(d, f):
    _, ext = os.path.splitext(f)
    if ext in whitelist_ext:
      return True
    return False

  containing_folder = get_containing_folder()
  containing_folder_basename = get_containing_folder_basename(containing_folder)
  zip_file_path = os.path.join(tempfile.mkdtemp(),
                               '{}.zip'.format(containing_folder_basename))
  with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as ziph:
    for root, dirs, files in os.walk(containing_folder):
      for file in files:
        if _should_zip(root, file):
          f_path = os.path.join(root, file)
          arcname = containing_folder_basename + f_path[len(containing_folder):]
          ziph.write(f_path, arcname=arcname)
  return zip_file_path


def zip_folder_relpath(folder, whitelist_ext=('.py', '.csv')):
  """ recursively zip input folder in relative path

  For example,
  "/Users/aaa/code/AwesomePackage/*" will be zipped as "AwesomePackage/*"
  where folder refers to /Users/aaa/code/AwesomePackage
  """
  blacklist_file = ['.DS_Store']
  blacklist_ext = ['pyc']
  blacklist_dir = ['.git', '.idea', 'TLeague.egg-info', '__pycache__']

  def _should_skip(d, f=''):
    dd = os.path.basename(d)
    if dd in blacklist_dir:
      return True
    if f in blacklist_file:
      return True
    _, ext = os.path.splitext(f)
    if ext in blacklist_ext:
      return True
    return False

  def _should_zip(d, f):
    _, ext = os.path.splitext(f)
    if ext in whitelist_ext:
      return True
    return False

  basename = os.path.basename(folder)
  zip_file_path = os.path.join(tempfile.mkdtemp(), '{}.zip'.format(basename))
  containing_folder = os.path.dirname(folder)
  with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as ziph:
    for root, dirs, files in os.walk(folder):
      for file in files:
        if _should_zip(root, file):
          f_path = os.path.join(root, file)
          arcname = basename + f_path[len(folder):]
          ziph.write(f_path, arcname=arcname)
  return zip_file_path


def prepare_working_folder_remote(local_zip_path, remote_working_folder,
                                  ip, port, username, password, overwrite=True,
                                  remote_worker_pre_cmd="", ssh_timeout=10):

  def _remote_target_exist(rem_tar):
    try:
      dummy_stat = sftp.stat(rem_tar)
      return True
    except IOError:
      return False

  zip_basename = path.basename(local_zip_path)
  remote_basename, _dummy_ext = path.splitext(zip_basename)
  remote_target = path.join(remote_working_folder, remote_basename)
  with paramiko.SSHClient() as ssh:
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, port=port, username=username, password=password,
                timeout=ssh_timeout)
    # make sure the target folder is ready
    sftp = ssh.open_sftp()
    is_remote_target_exist = _remote_target_exist(remote_target)
    print('on remote: {}, overwrite_remote: {}, remote_target_exist: {}'.format(
      ip, overwrite, is_remote_target_exist
    ))
    if overwrite or not is_remote_target_exist:
      # delete remote target and zip
      remote_zip_path = path.join(remote_working_folder, zip_basename)
      print('deleting {}:{}'.format(ip, remote_target))
      print('deleting {}:{}'.format(ip, remote_zip_path))
      run_command_remote_ssh(ssh, 'rm -r {} && rm {}'.format(remote_target,
                                                             remote_zip_path))
      # copy local zip to remote
      print('copying {} to {}:{}...'.format(local_zip_path, ip,
                                            remote_zip_path))
      sftp.chdir(remote_working_folder)
      sftp.put(local_zip_path, remote_zip_path)
      sleep(0.00026) # sftp actually blocks, doesn't need sleep at all
      # unzip
      print('extracting zip file on remote {} ...'.format(ip))
      run_command_remote_ssh(ssh, 'cd {} && unzip {}'.format(
        remote_working_folder, os.path.basename(local_zip_path)))
    else:
      print('not copying to remote ' )

    # perform pip install
    print('cd to {} and do pip installing...'.format(remote_target))
    tmp_cmd = remote_worker_pre_cmd + 'cd {} && pip install -e .'.format(
      remote_target
    )
    run_command_remote_ssh(ssh, tmp_cmd)
    print('done preparing working folder remote')


def get_ip():
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  try:
    # doesn't even have to be reachable
    s.connect(('xx.xx.xx.xx', 1))
    ip = s.getsockname()[0]
  except:
    ip = None
  finally:
    s.close()
  return ip


def is_remote_ip(ip_str):
  all_local_ips = ['localhost', '127.0.0.1']
  local_real_ip = get_ip()
  if local_real_ip is not None:
    all_local_ips.append(local_real_ip)

  return not ip_str in all_local_ips


def make_quote_cmd(cmd_str):
  # tmp = cmd_str.lstrip('\"').rstrip('\"')
  # return '\"' + tmp + '\"'
  return shlex.quote(cmd_str)


def make_unquote_cmd(cmd_str):
  return cmd_str.lstrip('\"').rstrip('\"')


def to_cmd_str(cmds):
  if isinstance(cmds, (list, tuple)):
    #cmds = " ".join(shlex_quote(str(v)) for v in cmds)
    cmds = ' '.join(str(v) for v in cmds)
  return cmds


def run_cmd_local(cmd, info='', pre_cmd=''):
  print('running command local')
  # pre_cmd = make_unquote_cmd(pre_cmd)
  print(pre_cmd + cmd)
  os.system(pre_cmd + cmd)
  print('done running command local \n')


def run_command_remote_ssh(handle_ssh, cmd):
  """ synchronously execute command remotely  """
  stdin, stdout, stderr = handle_ssh.exec_command(cmd)
  out, err = stdout.read(), stderr.read()
  pass


def run_cmd_remote(cmd, ip, port, username, password, info='', pre_cmd='',
                   ssh_timeout=10):
  print('running command remote')
  # pre_cmd = make_unquote_cmd(pre_cmd)
  with paramiko.SSHClient() as ssh:
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, port=port, username=username, password=password,
                timeout=ssh_timeout)
    print(pre_cmd + cmd)
    run_command_remote_ssh(ssh, pre_cmd + cmd)
  print('done sending command remote \n')


def create_tmux_cmd(sess_name, python_bin, new_win_name, cmd_exe, pre_cmd_exe):
  # make sure the command has been quoted
  safe_cmd = make_quote_cmd(pre_cmd_exe + cmd_exe)
  cmd = [
    python_bin, '-m',
    'tleague.scripts.send_tmux_cmd',
    '--sess_name', sess_name,
    '--new_win_name', new_win_name,
    '--cmd', safe_cmd
  ]
  return to_cmd_str(cmd)


def _func_run_cmd(_worker_cmd, zip_file_path, q_failed_workers,
                  remote_working_folder, overwrite_remote,
                  remote_worker_pre_cmd, local_worker_pre_cmd,
                  python_bin, tmux_sess, workers):
  # all workers must be on the same machines with the same ip
  # NOTE: overwrite inside the loop even only one ip is expected,
  # as the ip can be a virtual machine so that the same ip, usr_name, password
  # but different ssh_port leads to a different FileSystem!!
  for w in workers:
    worker_cmd = _worker_cmd(w)
    try:
      if is_remote_ip(w.ip):
        if overwrite_remote:
          prepare_working_folder_remote(
            zip_file_path, remote_working_folder, ip=w.ip,
            port=w.ssh_port,
            username=w.ssh_username, password=w.ssh_password,
            overwrite=overwrite_remote,
            remote_worker_pre_cmd=remote_worker_pre_cmd,
          )
        # To this extent, remote is prepared to run the command.
        tmux_cmd = create_tmux_cmd(
          sess_name=tmux_sess, python_bin=python_bin, new_win_name=w.job,
          cmd_exe=worker_cmd, pre_cmd_exe=remote_worker_pre_cmd
        )
        sleep(0.0015)  # the command actually blocks, doesn't need sleep at all!
        run_cmd_remote(cmd=tmux_cmd, ip=w.ip, port=w.ssh_port,
                       username=w.ssh_username, password=w.ssh_password,
                       info=worker_cmd, pre_cmd=remote_worker_pre_cmd)
        sleep(0.0015)  # the command actually blocks, doesn't need sleep at all!
      else:
        tmux_cmd = create_tmux_cmd(
          sess_name=tmux_sess, python_bin=python_bin, new_win_name=w.job,
          cmd_exe=worker_cmd, pre_cmd_exe=local_worker_pre_cmd
        )
        run_cmd_local(cmd=tmux_cmd, info=worker_cmd,
                      pre_cmd=local_worker_pre_cmd)
        sleep(0.3)
    except Exception as e:
      q_failed_workers.put((w, e))


def extrac_learner_spec(learners):
  """ extract learner specification that keeps the learner's order
  Returned example:
  ips = [xx.xx.xx.xx, xx.xx.xx.xx]
  ip_gpu_ports_list = [[0:1004:1005], [0:1004:1005, 1:1006:1007]]
  which means 2 hosts,
  host1 with 1 process using gpu 0 and port 1004:1005, host2 with 2 processes
  using gpu 1 and port 1004:1005, gpu 2 and port 1006:1007
  """
  ips = []
  gpu_ports_list = []
  for w in learners:
    gpu_ports = ':'.join([w.cuda_visible_devices, w.port1, w.port2])
    if w.ip not in ips:
      ips.append(w.ip)
      gpu_ports_list.append([gpu_ports])
    else:
      gpu_ports_list[ips.index(w.ip)].append(gpu_ports)
  return ips, gpu_ports_list
