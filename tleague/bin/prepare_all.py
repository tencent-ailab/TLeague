from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from time import sleep
from multiprocessing import Pool
import multiprocessing
from functools import partial
from os import path

from absl import app
from absl import flags

from . import parse_workers_spec_csv
from . import group_workers_by_ip
from . import is_remote_ip
from . import zip_containing_folder_relpath
from . import zip_folder_relpath
from . import prepare_working_folder_remote


FLAGS = flags.FLAGS
flags.DEFINE_string('clust_spec_csv_path', '../sandbox/clust_spec_example.csv',
                    'cluster specification csv file path.')
flags.DEFINE_string('remote_working_folder', '/root/code',
                    'unified remote working folder across all machines.')
flags.DEFINE_boolean('force_overwrite_remote', True,
                     'Should force overwriting remote working folder?')
flags.DEFINE_string('remote_worker_pre_cmd', '',
                    "pre command for each REMOTE worker. "
                    "e.g., activate python virutal env")
flags.DEFINE_multi_string(
  "prepare_python_package_path", [],
  "pip installable package path in comma separated format: "
  "package_local_fullpath,remote_fullpath "
  "It can occur multiple times. E.g., \n"
  "-p /root/Arena,/root/Arena\n"
  "-p /home/me/pysc2,/home/work/pysc2\n"
  "...",
  short_name='p'
)
flags.DEFINE_integer('n_process', 1,
                     'number of parallel processes for connections.')


def _parse_packages(args):

  def _known_package_ext_whitelist(package_basename):
    if package_basename.lower() == 'pysc2':  # Tencent PySC2 Extension
      return ['.py', '.serialized', '.SC2Map', '.md']
    if package_basename.lower() == 'arena':
      return ['.py', '.md', '.wad', '.cfg']
    return ['.py', '.md', '.csv']  # default to these typical files

  packages = []
  for item in args:
    local_p, remote_p = item.split(',')
    basename = path.basename(local_p)
    local_dir = path.dirname(local_p)
    remote_dir = path.dirname(remote_p)
    # check is dir
    if not path.isdir(local_p):
      raise ValueError('{} is not a folder'.format(local_p))
    # check name consistence
    if basename != path.basename(remote_p):
      raise ValueError(
        'remote basename {} must equal to local basename {}'.format(
        path.basename(remote_p), basename)
      )
    # zip file
    whitelist_ext = _known_package_ext_whitelist(basename)
    zip_filepath = zip_folder_relpath(local_p, whitelist_ext=whitelist_ext)
    # book-keep it
    packages.append({'local_dir': local_dir, 'remote_dir': remote_dir,
                     'basename': basename, 'zip_filepath': zip_filepath})
  return packages


def _func_run_cmd(zip_file_path, packages, q_failed_workers, workers):
  # all workers must be on the same machines with the same ip
  # NOTE: overwrite inside the loop even only one ip is expected,
  # as the ip can be a virtual machine so that the same ip, usr_name, password
  # but different ssh_port leads to a different FileSystem!!
  overwrite_remote = FLAGS.force_overwrite_remote
  for w in workers:
    try:
      if is_remote_ip(w.ip):
        prepare_working_folder_remote(
          zip_file_path, FLAGS.remote_working_folder, ip=w.ip,
          port=w.ssh_port,
          username=w.ssh_username, password=w.ssh_password,
          overwrite=overwrite_remote,
          remote_worker_pre_cmd=FLAGS.remote_worker_pre_cmd
        )
        # the command actually blocks, doesn't need sleep at all!
        # Dare you delete it though??
        sleep(0.00015)
        for package in packages:
          prepare_working_folder_remote(
            package['zip_filepath'], package['remote_dir'], ip=w.ip,
            port=w.ssh_port,
            username=w.ssh_username, password=w.ssh_password,
            overwrite=overwrite_remote,
            remote_worker_pre_cmd=FLAGS.remote_worker_pre_cmd
          )
    except Exception as e:
      q_failed_workers.put((w, e))


def main(_):
  workers = parse_workers_spec_csv(FLAGS.clust_spec_csv_path)
  ip_to_workers = group_workers_by_ip(workers)

  zip_file_path = zip_containing_folder_relpath()
  packages = _parse_packages(FLAGS.prepare_python_package_path)

  q_failed_workers = multiprocessing.Manager().Queue()
  mp = Pool(processes=FLAGS.n_process)
  tmp_func = partial(_func_run_cmd, zip_file_path, packages, q_failed_workers)
  mp.map(tmp_func, list(ip_to_workers.values()))
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
