from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import uuid
from multiprocessing import Process

import timeout_decorator
from tleague.learners.learner_apis import LearnerAPIs
from tleague.learners.base_learner import BaseLearner
from tleague.model_pools.model_pool import ModelPool
from tleague.league_mgrs.league_mgr import LeagueMgr


class TestActorCommunicator(unittest.TestCase):

  def setUp(self):
    self._model_process = Process(
        target=lambda: ModelPool(ports="11001:11006").run())
    self._league_process = Process(
        target=lambda: LeagueMgr(
            port=11007, model_pool_addrs=["localhost:11001:11006"],
            mutable_hyperparam_type='MutableHyperparam').run())

    self._model_process.start()
    self._league_process.start()

  def tearDown(self):
    self._model_process.terminate()
    self._league_process.terminate()

  @timeout_decorator.timeout(2)
  def test_request_learner_id(self):
    self.learner_id = str(uuid.uuid1())
    self.learner = BaseLearner(league_mgr_addr="localhost:11007",
                               model_pool_addrs=["localhost:11001:11006"],
                               learner_ports=[10010, 10011],
                               learner_id=self.learner_id)
    self.learner_task = self.learner._request_task()
    self.learner._notify_task_begin(self.learner_task)
    self.learner._model_pool_apis.push_model(None, self.learner_task.hyperparam,
                                             self.learner_task.model_key)
    learner_api = LearnerAPIs(learner_addr="localhost:10010:10011")
    learner_id = learner_api.request_learner_id()
    self.assertEqual(learner_id, self.learner_id)
    data_sent = "any python object"
    learner_api.push_data(data_sent)
    data_recv = self.learner._pull_data()
    self.assertEqual(data_recv, data_sent)


if __name__ == '__main__':
  unittest.main()
