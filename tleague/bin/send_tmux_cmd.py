""" send command to tmux window """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tempfile
import os

from absl import app
from absl import flags
import libtmux


FLAGS = flags.FLAGS
flags.DEFINE_string('sess_name', 'tlea', 'tmux session name for each machine')
flags.DEFINE_string('new_win_name', 'new_win', 'tmux new window name')
flags.DEFINE_string('cmd', '\"echo hello world\"',
                    'command to be sent to tmux. quote it if spaces.')


def _long_cmd_to_tmp_file(cmd_str):
  fd, file_path = tempfile.mkstemp(suffix='.sh')
  with os.fdopen(fd, "w") as f:
    f.write(cmd_str)
  return file_path


def main(_):
  tmux_sess_name = FLAGS.sess_name
  tmux_win_name = FLAGS.new_win_name
  cmd_str = FLAGS.cmd.lstrip('\"').rstrip('\"')

  print('sending command to tmux sess {}'.format(tmux_sess_name))
  print(cmd_str)

  # find or create the session
  tmux_server = libtmux.Server()
  tmux_sess = None
  try:
    tmux_sess = tmux_server.find_where({'session_name': tmux_sess_name})
  except:
    pass
  if tmux_sess is None:
    tmux_sess = tmux_server.new_session(tmux_sess_name)
  # create new window/pane, get it and send the command
  tmux_sess.new_window(window_name=tmux_win_name)
  pane = tmux_sess.windows[-1].panes[0]
  # run the command
  if len(cmd_str) < 512:
    pane.send_keys(cmd_str, suppress_history=False)
  else:
    # tmux may reject too long command
    # so let's write it to a temp file, and run it in tmux
    tmp_file_path = _long_cmd_to_tmp_file(cmd_str)
    tmp_cmd_str = ["cat {}".format(tmp_file_path),
                   "sh {}".format(tmp_file_path)]
    pane.send_keys("\n".join(tmp_cmd_str), suppress_history=False)
    #pos.unlink(tmp_file_path)

  print("done sending command to tmux.")


if __name__ == '__main__':
  app.run(main)
