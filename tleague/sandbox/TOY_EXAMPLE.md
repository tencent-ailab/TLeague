## Usage: A Toy Example

Deploy TLeague:

* GPU-Machine(192.1.1.1):
	* Learner-A (10001, 10002), Learner-B (10003, 10004)
	* LeagueMgr (10005), Model-A (10006, 10007)
* GPU-Machine(192.1.1.2):
	* Learner-C (10001, 10002)
	* Model-B (10003, 10004)
* CPU-Machine(1.1.1.1):
	* Actor-A (Learner-A), Actor-B (Learner-B)
* CPU-Machine(1.1.1.2):
	* Actor-C (Learner-C), Actor-D (For Eval)

### Deployment

* GPU-Machine (192.1.1.1):

~~~~
learner = PPOLearner(league_mgr_addr="192.1.1.1:10005",
                     model_pool_addrs=["192.1.1.1:10006:10007",
                                       "192.1.1.2:10003:10004"],
                     learner_ports="10001:10002")
learner.run()
~~~~

~~~~
learner = PPOLearner(league_mgr_addr="192.1.1.1:10005",
                     model_pool_addrs=["192.1.1.1:10006:10007",
                                       "192.1.1.2:10003:10004"],
                     learner_ports="10003:10004")
learner.run()
~~~~

~~~~
league_mgr = LeagueManager(port="10005",
                           model_pool_addrs=["192.1.1.1:10006:10007",
                                             "192.1.1.2:10003:10004"])
league_mgr.run()
~~~~

~~~~
model_pool = ModelPool(ports="10006:10007")
model_pool.run()
~~~~

* GPU-Machine (192.1.1.2):

~~~~
learner = PPOLearner(league_mgr_addr="192.1.1.1:10005",
                     model_pool_addrs=["192.1.1.1:10006:10007",
                                       "192.1.1.2:10003:10004"],
                     learner_ports="10001:10002")
learner.run()
~~~~

~~~~
model_pool = ModelPool(ports="10003:10004")
model_pool.run()
~~~~

* CPU-Machine(1.1.1.1):

~~~~
actor = PPOActor(league_mgr_addr="192.1.1.1:10005",
                 model_pool_addrs=["192.1.1.1:10006:10007"
                                   "192.1.1.2:10003:10004"],
                 learner_addr="192.1.1.1:10001:10002")
actor.run()
~~~~
~~~~
actor = PPOActor(league_mgr_addr="192.1.1.1:10005",
                 model_pool_addrs=["192.1.1.1:10006:10007",
                                   "192.1.1.2:10003:10004"],
                 learner_addr="192.1.1.1:10003:10004")
actor.run()
~~~~

* CPU-Machine(1.1.1.2):

~~~~
actor = PPOActor(league_mgr_addr="192.1.1.1:10005",
                 model_pool_addrs=["192.1.1.1:10006:10007",
                                   "192.1.1.2:10003:10004"],
                 learner_addr="192.1.1.2:10001:10002")
actor.run()
~~~~
~~~~
actor = PPOActor(league_mgr_addr="192.1.1.1:10005",
                 model_pool_addrs=["192.1.1.1:10006:10007",
                                   "192.1.1.2:10003:10004"],
                 learner_addr=None)
actor.run()
~~~~