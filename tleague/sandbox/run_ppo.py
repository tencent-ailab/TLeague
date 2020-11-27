from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from tleague.actors.ppo_actor import PPOActor
from tleague.learners.ppo_learner3 import PPOLearner

FLAGS = flags.FLAGS
flags.DEFINE_string("league_mgr_addr", "localhost:10005",
                    "League manager address.")
flags.DEFINE_string("model_pool_addrs", "localhost:10006,localhost:10007",
                    "Model Pool addresses")
flags.DEFINE_string("learner_addrs", "localhost:10003,localhost:10004",
                    "Learner addresses")
flags.DEFINE_string("actor_addrs",
                    "localhost:10000,localhost:10000,localhost:10000",
                    "Actor addresses")


def start_learner():
  learner_ports = [item[1] for item in FLAGS.learner_ports.split(',')]
  learner = PPOLearner(league_mgr_addr=FLAGS.league_mgr_addr,
                       model_pool_addrs=FLAGS.model_pool_addrs.split(','),
                       learner_ports=learner_ports)
  learner.run()


def start_actor():
  actor = PPOActor(league_mgr_addr=FLAGS.league_mgr_addr,
                   model_pool_addrs=FLAGS.model_pool_addrs.split(','),
                   learner_addr=FLAGS.learner_addr.split(','))
  actor.run()


def main(_):
  learner = PPOLearner(league_mgr_addr=FLAGS.league_mgr_addr,
                       model_pool_addrs=FLAGS.model_pool_addrs.split(','),
                       learner_ports=FLAGS.learner_ports.split(','))
  learner.run()


def main_single_old():
  workers = parse_workers_spec_csv(FLAGS.clust_spec_csv_path)
  job_to_workers = split_workers(workers)
  act_addr_to_lrn = pair_actors_learners(actors=job_to_workers['actor'],
                                         learners=job_to_workers['learner'])
  zip_file_path = zip_containing_folder_relpath()

  failed_workers = []
  ip_to_remote_overwrite = {}
  for w in workers:
    worker_cmd = create_worker_cmd(w, job_to_workers, act_addr_to_lrn)
    try:
      if is_remote_ip(w.ip):
        tmux_cmd = create_tmux_cmd(
          sess_name=FLAGS.tmux_sess, new_win_name=w.job, cmd_exe=worker_cmd,
          pre_cmd_exe=FLAGS.remote_worker_pre_cmd
        )
        if ip_to_remote_overwrite.get(w.ip, None) is None:
          ip_to_remote_overwrite[w.ip] = FLAGS.force_overwrite_remote
        prepare_working_folder_remote(
          zip_file_path, FLAGS.remote_working_folder, ip=w.ip, port=w.ssh_port,
          username=w.ssh_username, password=w.ssh_password,
          overwrite=ip_to_remote_overwrite[w.ip]
        )
        ip_to_remote_overwrite[w.ip] = False  # one ip only overwrites once!
        sleep(0.75)
        run_cmd_remote(cmd=tmux_cmd, ip=w.ip, port=w.ssh_port,
                      username=w.ssh_username, password=w.ssh_password,
                      info=worker_cmd, pre_cmd=FLAGS.remote_worker_pre_cmd)
        sleep(0.15)
      else:
        tmux_cmd = create_tmux_cmd(
          sess_name=FLAGS.tmux_sess, new_win_name=w.job, cmd_exe=worker_cmd,
          pre_cmd_exe=FLAGS.local_worker_pre_cmd
        )
        run_cmd_local(cmd=tmux_cmd, info=worker_cmd,
                      pre_cmd=FLAGS.local_worker_pre_cmd)
        sleep(0.3)
    except Exception as e:
      failed_workers.append((w, e))

  print('\nall commands sent. Succeeded: {}, Failed: {}'.format(
    len(workers) - len(failed_workers), len(failed_workers)
  ))
  for w, e in failed_workers:
    print(w)
    print('{}: {}'.format(type(e), e))


if __name__ == '__main__':
  app.run(main)
