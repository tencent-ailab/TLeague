""" script that reads in the customized ip_list file and outputs the CSV
needed by run_all_xxx.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
from collections import namedtuple

from absl import app
from absl import flags

from tleague.scripts import WorkerSpec


FLAGS = flags.FLAGS
flags.DEFINE_string('input_path', '../sandbox/tail32',
                    'input ip_list file path')
flags.DEFINE_string('base_port', '9999',
                    'base port for port1 & port2. '
                    'It avoids port conflict on the same machine')
flags.DEFINE_string('output_path', '../sandbox/b70_tail32.csv',
                    'output csv file path, the cluster spec.')
flags.DEFINE_integer('num_actor', 1, 'number of actors on each machine')
flags.DEFINE_integer('num_actor_eval', 0, 'number of eval actors on each machine')


Worker = namedtuple('Worker', 'ip ssh_username ssh_port ssh_password')


def write_cluster_csv(workers, cluster_csv_path):
  ip_to_cur_port = {}
  with open(cluster_csv_path, 'w') as fp:
    csv_writer = csv.writer(fp, delimiter=',', quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)
    # write header
    csv_writer.writerow(list(WorkerSpec._fields))
    # write body
    for worker in workers:
      # avoid ip conflict on the same machine (ip)
      if ip_to_cur_port.get(worker.ip, None) is None:
        ip_to_cur_port[worker.ip] = FLAGS.base_port
      for _ in range(FLAGS.num_actor):
        item = WorkerSpec(
          job='actor',
          ip=worker.ip,
          port1=ip_to_cur_port[worker.ip],
          port2=str(int(ip_to_cur_port[worker.ip]) + 1),
          cuda_visible_devices='',
          ssh_username=worker.ssh_username,
          ssh_password=worker.ssh_password,
          ssh_port=worker.ssh_port
        )
        ip_to_cur_port[worker.ip] = str(int(ip_to_cur_port[worker.ip]) + 2)
        csv_writer.writerow(list(item))

    for worker in workers:
      for _ in range(FLAGS.num_actor_eval):
        item = WorkerSpec(
          job='actor_eval',
          ip=worker.ip,
          port1=ip_to_cur_port[worker.ip],
          port2=str(int(ip_to_cur_port[worker.ip]) + 1),
          cuda_visible_devices='',
          ssh_username=worker.ssh_username,
          ssh_password=worker.ssh_password,
          ssh_port=worker.ssh_port
        )
        ip_to_cur_port[worker.ip] = str(int(ip_to_cur_port[worker.ip]) + 2)
        csv_writer.writerow(list(item))


def read_ip_list(ip_list_path):
  with open(ip_list_path, 'r') as fp:
    workers = [Worker(*line.strip().split(' '))
               for line in fp.readlines()]
  return workers


def main(_):
  workers = read_ip_list(FLAGS.input_path)
  write_cluster_csv(workers, FLAGS.output_path)


if __name__ == '__main__':
  app.run(main)
