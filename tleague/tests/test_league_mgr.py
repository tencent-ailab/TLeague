from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import unittest
import uuid
from multiprocessing import Process

import timeout_decorator
from tleague.utils.tl_types import ActorTask, LearnerTask
from tleague.league_mgrs.league_mgr_apis import LeagueMgrAPIs
from tleague.hyperparam_mgr.hyperparam_types import MutableHyperparam
from tleague.league_mgrs.league_mgr import LeagueMgr
from tleague.model_pools.model import Model
from tleague.model_pools.model_pool import ModelPool
from tleague.model_pools.model_pool_apis import ModelPoolAPIs
from tleague.utils.tl_types import MatchResult


class TestLeagueManager(unittest.TestCase):

  def setUp(self):
    self._model_process = Process(
        target=lambda: ModelPool(ports="11001:11006").run())
    self._model_process2 = Process(
        target=lambda: ModelPool(ports="11011:11016").run())
    self._league_process = Process(
        target=lambda: LeagueMgr(
            port="11007",
            model_pool_addrs=["localhost:11001:11006"],
            mutable_hyperparam_type='MutableHyperparam',
            save_checkpoint_root="./checkpoints",
            save_interval_secs=3).run())
    self._model_process.start()
    self._model_process2.start()
    self._league_process.start()

  def tearDown(self):
    self._model_process.terminate()
    self._model_process2.terminate()
    self._league_process.terminate()

  @timeout_decorator.timeout(2)
  def test_learner_task(self):
    league_client = LeagueMgrAPIs(league_mgr_addr="localhost:11007")
    learner_id = str(uuid.uuid1())
    task = league_client.request_learner_task(learner_id=learner_id)
    self.assertTrue(isinstance(task, LearnerTask))
    query_task = league_client.query_learner_task(learner_id=learner_id)
    self.assertEqual(task.model_key, query_task.model_key)
    self.assertEqual(task.parent_model_key, query_task.parent_model_key)
    league_client.notify_learner_task_begin(learner_id=learner_id,
                                            learner_task=task)
    league_client.notify_learner_task_end(learner_id=learner_id)


  @timeout_decorator.timeout(2)
  def test_actor_task(self):
    actor_id = str(uuid.uuid1())
    learner_id = str(uuid.uuid1())
    league_client = LeagueMgrAPIs(league_mgr_addr="localhost:11007")
    learner_task = league_client.request_learner_task(learner_id=learner_id)
    league_client.notify_learner_task_begin(learner_id=learner_id,
                                            learner_task=learner_task)

    model_client = ModelPoolAPIs(model_pool_addrs=["localhost:11001:11006"])
    hyperparam = MutableHyperparam()
    model_client.push_model(None, hyperparam, str(uuid.uuid1()))

    task = league_client.request_actor_task(actor_id=actor_id,
                                            learner_id=learner_id)
    self.assertTrue(isinstance(task, ActorTask))
    league_client.notify_actor_task_begin(actor_id=actor_id)
    league_client.notify_actor_task_end(
      actor_id=actor_id,
      match_result=MatchResult(task.model_key1, task.model_key2, 1))

  @timeout_decorator.timeout(20)
  def test_checkpoint(self):
    league_client = LeagueMgrAPIs(league_mgr_addr="localhost:11007")
    model_client1 = ModelPoolAPIs(model_pool_addrs=["localhost:11001:11006"])
    hyperparam = MutableHyperparam()
    model_key1 = str(uuid.uuid1())
    model_key2 = str(uuid.uuid1())
    model_client1.push_model("model_data1", hyperparam, model_key1)
    model_client1.push_model("model_data2", hyperparam, model_key2)
    time.sleep(4)
    league_client.request_add_model(
      Model("model_data1", hyperparam, model_key1))
    model_client1.push_model("model_data3", hyperparam, model_key2)
    time.sleep(3)
    checkpoints = [filename for filename in os.listdir("./checkpoints")
                   if filename.startswith("checkpoint")]
    self.assertTrue(len(checkpoints) > 0)

    checkpoint_dir = os.path.join("./checkpoints", checkpoints[-1])
    league_process = Process(
        target=lambda: LeagueMgr(
          port="11008",
          model_pool_addrs=["localhost:11011:11016"],
          mutable_hyperparam_type='MutableHyperparam',
          restore_checkpoint_dir=checkpoint_dir).run())
    league_process.start()

    model_client2 = ModelPoolAPIs(model_pool_addrs=["localhost:11011:11016"])
    time.sleep(2)
    keys = model_client2.pull_keys()
    self.assertTrue(model_key1 in keys)
    self.assertTrue(model_key2 in keys)
    model1 = model_client1.pull_model(model_key1)
    model2 = model_client2.pull_model(model_key1)
    self.assertEqual(model1.model, model2.model)
    self.assertEqual(model1.key, model2.key)
    self.assertEqual(model1.createtime, model2.createtime)
    model1 = model_client1.pull_model(model_key2)
    model2 = model_client2.pull_model(model_key2)
    self.assertEqual(model1.model, model2.model)
    self.assertEqual(model1.key, model2.key)
    self.assertEqual(model1.createtime, model2.createtime)
    league_process.terminate()
    #shutil.rmtree('./checkpoints')


if __name__ == '__main__':
  unittest.main()
