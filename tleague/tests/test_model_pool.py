from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import timeout_decorator
from multiprocessing import Process
import uuid


from tleague.model_pools.model_pool import ModelPool
from tleague.model_pools.model_pool_apis import ModelPoolAPIs


class TestModelPoolClient(unittest.TestCase):

  def setUp(self):
    self._server_process1 = Process(
        target=lambda: ModelPool(ports="11001:11006").run())
    self._server_process2 = Process(
        target=lambda: ModelPool(ports="11002:11007").run())
    self._server_process1.start()
    self._server_process2.start()

  def tearDown(self):
    self._server_process1.terminate()
    self._server_process2.terminate()

  @timeout_decorator.timeout(1)
  def test_push_model(self):
    client = ModelPoolAPIs(model_pool_addrs=["localhost:11001:11006",
                                             "localhost:11002:11007"])
    key1 = str(uuid.uuid1())
    client.push_model(None, None, key1)

  @timeout_decorator.timeout(1)
  def test_pull_keys(self):
    client = ModelPoolAPIs(model_pool_addrs=["localhost:11001:11006",
                                             "localhost:11002:11007"])
    key1 = str(uuid.uuid1())
    key2 = str(uuid.uuid1())
    client.push_model(None, None, key1)
    client.push_model(None, None, key2)
    client.push_model(None, None, key1)

    saved_keys = client.pull_keys()
    self.assertEqual(len(saved_keys), 2)
    self.assertTrue(key1 in saved_keys)
    self.assertTrue(key2 in saved_keys)

  @timeout_decorator.timeout(1)
  def test_pull_model(self):
    client = ModelPoolAPIs(model_pool_addrs=["localhost:11001:11006",
                                             "localhost:11002:11007"])
    key1 = str(uuid.uuid1())
    key2 = str(uuid.uuid1())
    client.push_model("any_model_object", None, key1)
    client.push_model("any_model_object", None, key2)
    client.push_model("updated_model_object", None, key2)

    model1 = client.pull_model(key1)
    self.assertEqual(model1.model, "any_model_object")
    model2 = client.pull_model(key2)
    self.assertEqual(model2.model, "updated_model_object")

  @timeout_decorator.timeout(1)
  def test_pull_hyperparam(self):
    client = ModelPoolAPIs(model_pool_addrs=["localhost:11001:11006",
                                             "localhost:11002:11007"])
    key1 = str(uuid.uuid1())
    key2 = str(uuid.uuid1())
    client.push_model(None, "any_hyperparam_object", key1)
    client.push_model(None, "any_hyperparam_object", key2)
    client.push_model(None, "updated_hyperparam_object", key2)

    hyperparam1 = client.pull_attr('hyperparam', key1)
    self.assertEqual(hyperparam1, "any_hyperparam_object")
    hyperparam2 = client.pull_attr('hyperparam', key2)
    self.assertEqual(hyperparam2, "updated_hyperparam_object")


if __name__ == '__main__':
  unittest.main()
