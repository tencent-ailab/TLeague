import pickle
import zmq
import random
from tleague.utils.io import TensorZipper


class InfServerAPIs(object):
  def __init__(self, inference_server_addr, ds, compress=False, timeout=30000):
    self._zmq_context = zmq.Context()
    ip_ports = list(inference_server_addr.split(','))
    self.server_addr = inference_server_addr
    random.shuffle(ip_ports)
    self._ip_ports = ip_ports
    self.ds = ds
    self._compress = compress
    self.timeout = timeout
    self._req_socket = None
    self._rebuild_socket()

  def _rebuild_socket(self):
    if self._req_socket is not None:
      self._req_socket.setsockopt(zmq.LINGER, 0)
      self._req_socket.close()
    self._req_socket = self._zmq_context.socket(zmq.REQ)
    self._req_socket.setsockopt(zmq.RCVTIMEO, self.timeout)
    for ip_port in self._ip_ports:
      self._req_socket.connect("tcp://{}".format(ip_port))

  def request_output(self, obs):
    # obs must match the data structure defined in self.ds
    data = self.ds.flatten(obs)
    if self._compress:
      data = TensorZipper.compress(data)
    else:
      data = pickle.dumps(data)
    self._req_socket.send(data)
    while True:
      try:
        ret = self._req_socket.recv_pyobj()
        break
      except Exception as e:
        print(f'Exception:{e} After {self.timeout} ms for request inference '
              f'service {self.server_addr}, restart a socket and try again!')
        self._rebuild_socket()
        self._req_socket.send(data)
    return ret
