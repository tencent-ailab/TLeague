import pickle
import zmq
import random
from tleague.utils.io import TensorZipper


class InfServerAPIs(object):
  def __init__(self, inference_server_addr, ds, compress=False):
    self._zmq_context = zmq.Context()
    self._req_socket = self._zmq_context.socket(zmq.REQ)
    ip_ports = list(inference_server_addr.split(','))
    random.shuffle(ip_ports)
    for ip_port in ip_ports:
      self._req_socket.connect("tcp://{}".format(ip_port))
    self.ds = ds
    self._compress = compress

  def request_output(self, obs):
    # obs must match the data structure defined in self.ds
    data = self.ds.flatten(obs)
    if self._compress:
      data = TensorZipper.compress(data)
    else:
      data = pickle.dumps(data)
    self._req_socket.send(data)
    return self._req_socket.recv_pyobj()
