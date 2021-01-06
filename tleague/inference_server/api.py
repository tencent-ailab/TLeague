import pickle
import zmq
import random
import time
from tleague.utils.io import TensorZipper


class InfServerAPIs(object):
  def __init__(self, inference_server_addr, ds, compress=False, timeout=30000):
    self._zmq_context = zmq.Context()
    ip_ports = list(inference_server_addr.split(','))
    self.server_addr = inference_server_addr
    self._ip_ports = ip_ports
    self.ds = ds
    self._compress = compress
    self.timeout = timeout
    self._req_sockets = []
    self._dead_sockets = {}
    self.poll = zmq.Poller()
    self._build_sockets()

  def _build_sockets(self):
    for ip_port in self._ip_ports:
      req_socket = self._zmq_context.socket(zmq.REQ)
      req_socket.connect("tcp://{}".format(ip_port))
      self._req_sockets.append(req_socket)
      self.poll.register(req_socket, zmq.POLLIN)

  def _rebuild_socket(self, socket):
    endpoint = str(socket.getsockopt(zmq.LAST_ENDPOINT), encoding="utf-8")
    if socket in self._dead_sockets:
      self._dead_sockets.pop(socket)
    else:
      self._req_sockets.remove(socket)
      print(f'After {self.timeout} ms for request inference '
            f'service {endpoint}, restart a socket!')
    self.poll.unregister(socket)
    socket.setsockopt(zmq.LINGER, 0)
    socket.close()
    req_socket = self._zmq_context.socket(zmq.REQ)
    req_socket.connect(endpoint)
    self.poll.register(req_socket, zmq.POLLIN)
    self._dead_sockets[req_socket] = time.time()
    return req_socket

  def _get_random_socket(self, data):
    while len(self._req_sockets) == 0:
      print(f'All the infservers {self.server_addr} are down!', flush=True)
      socks = dict(self.poll.poll(self.timeout))
      self._keep_dead_sockets(socks, data)
    return random.sample(self._req_sockets,1)[0]

  def _keep_dead_sockets(self, socks, data):
    for dead_socket in self._dead_sockets:
      if socks.get(dead_socket) == zmq.POLLIN:
        self._req_sockets.append(dead_socket)
        self._dead_sockets.pop(dead_socket)
    # keep push data every 2*self.timeout to dead sockets
    long_dead_sockets = [socket for socket in self._dead_sockets
                         if (time.time() - self._dead_sockets[socket]
                             > 2*self.timeout/1000.)]
    for socket in long_dead_sockets:
      new_socket = self._rebuild_socket(socket)
      new_socket.send(data)

  def _request(self, data):
    ret = None
    while ret is None:
      socket = self._get_random_socket(data)
      socket.send(data)
      socks = dict(self.poll.poll(self.timeout))
      if socks.get(socket) == zmq.POLLIN:
        ret = socket.recv_pyobj()
      else:
        self._rebuild_socket(socket)
      self._keep_dead_sockets(socks, data)
    return ret

  def request_output(self, obs):
    # obs must match the data structure defined in self.ds
    data = self.ds.flatten(obs)
    if self._compress:
      data = TensorZipper.compress(data)
    else:
      data = pickle.dumps(data)
    return self._request(data)
