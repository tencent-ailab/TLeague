import pickle
from tleague.utils import logger

def robust_pyobj_recv(socket):
    msg = socket.recv()
    try:
        # equivalent to recv_pyobj()
      msg = socket._deserialize(msg, pickle.loads)
    except:
      msg = None
    return msg

def robust_string_recv(socket):
    msg = socket.recv()
    try:
        # equivalent to recv_string()
      msg = socket._deserialize(msg, lambda buf: buf.decode('utf-8'))
    except:
      msg = None
    return msg