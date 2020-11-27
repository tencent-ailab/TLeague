import importlib
import socket
import time
import subprocess
import os

import psutil


def read_config_dict(config_name):
  try:
    cfg = _read_config_dict_py_module(config_name)
    print('successfully _read_config_dict_py_module {}'.format(config_name),
          flush=True)
    return cfg
  except:
    pass

  try:
    cfg = _read_config_dict_py_expression(config_name)
    print('successfully _read_config_dict_py_expression {}'.format(config_name),
          flush=True)
    return cfg
  except:
    pass

  if config_name == "":
    print("Empty config string, returning empty dict", flush=True)
    return {}

  raise ValueError('Unknown cfg {}'.format(config_name))


def _read_config_dict_py_module(config_name):
  config_name.lstrip("\"").rstrip("\"")
  config_module, config_name = config_name.rsplit(".", 1)
  config = getattr(importlib.import_module(config_module), config_name)
  return config


def _read_config_dict_py_expression(config_name):
  return eval(config_name)


def import_module_or_data(import_path):
  try:
    maybe_module, maybe_data_name = import_path.rsplit(".", 1)
    print('trying from module {} import data {}'.format(maybe_module,
                                                        maybe_data_name))
    return getattr(importlib.import_module(maybe_module), maybe_data_name)
  except Exception as e:
    print('Cannot import data from the module path, error {}'.format(str(e)))

  try:
    print('trying to import module {}'.format(import_path))
    return importlib.import_module(import_path)
  except Exception as e:
    print('Cannot import module, error {}'.format(str(e)))

  raise ImportError('Cannot import module or data using {}'.format(import_path))


def get_ip_hostname():
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  s.connect(("8.8.8.8", 80))
  return s.getsockname()[0], socket.gethostname()


def now():
  return '[' + time.strftime('%Y%m%d%H%M%S') + ']'


def kill_sc2_processes():
  """ kill all the sc2 processes, very dangerous.

  It kills ALL sc2 processes on the machine"""
  outputs = subprocess.check_output(
    "ps -aux | grep StarCraft | grep -v grep | awk '{print $2}' ",
    shell=True
  ).decode().strip('\n')
  if outputs:
    print("Killing all StarCraftII processes:\n{}".format(outputs))
    subprocess.check_output('kill -9 {}'.format(' '.join(outputs.split('\n'))),
                            shell=True)


def kill_sc2_processes_v2():
  """ kill all the children sc2 processes. """
  p_me = psutil.Process(os.getpid())
  print(p_me)
  for p in p_me.children():
    if p.name() == 'SC2':
      print('Killing {}'.format(p))
      p.kill()
      print('Killed {}'.format(p))
