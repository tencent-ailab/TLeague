from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import time

from tleague.hyperparam_mgr.hyperparam_mgr import HyperparamMgr
from tleague.league_mgrs.base_league_mgr import BaseLeagueMgr
from tleague.utils import logger
from tleague.utils.tl_types import ActorTask
from tleague.utils.tl_types import LearnerTask
from tleague.utils.tl_types import LeagueMgrErroMsg
from tleague.utils import now
from tleague.utils import import_module_or_data


class LeagueMgr(BaseLeagueMgr):
  """ League Manager that maintains multiple parallel learners. """

  def __init__(self,
               port,
               model_pool_addrs,
               mutable_hyperparam_type,
               hyperparam_config_name=None,
               restore_checkpoint_dir=None,
               save_checkpoint_root=None,
               save_interval_secs=3600,
               game_mgr_type='tleague.game_mgr.game_mgrs.RandomGameMgr',
               game_mgr_config=None,
               mute_actor_msg=False,
               verbose=0,
               init_model_paths=None,
               save_learner_meta=False):
    super(LeagueMgr, self).__init__(port,
                                    model_pool_addrs,
                                    restore_checkpoint_dir,
                                    save_checkpoint_root,
                                    save_interval_secs,
                                    mute_actor_msg,
                                    save_learner_meta,
                                    verbose=verbose)
    logger.set_level(verbose)
    logger.configure(dir='league_log/', format_strs=['stdout', 'log'])

    self._game_mgr_type = game_mgr_type
    game_mgr_cls = import_module_or_data(game_mgr_type)
    logger.log('__init__: game_mgr_type: {}'.format(game_mgr_type))
    game_mgr_config = game_mgr_config or {}
    game_mgr_config['pgn_file'] = (game_mgr_config.get('pgn_file', None)
                                   or 'example.pgn')
    game_mgr_config['verbose'] = (game_mgr_config.get('verbose', None)
                                  or verbose)
    logger.log('__init__: game_mgr_config: {}'.format(game_mgr_config))
    self.game_mgr = game_mgr_cls(**game_mgr_config)

    logger.log('__init__: hyperparam_mgr: {}, hyperparam_config: {}'.format(
      mutable_hyperparam_type, hyperparam_config_name
    ))
    self._hyper_mgr = HyperparamMgr(
        self._model_pool_apis, mutable_hyperparam_type, hyperparam_config_name)

    self.init_model_keys = []
    if init_model_paths is not None:
      assert isinstance(init_model_paths, list)
      logger.log('__init__: init_model from paths {}:'.format(init_model_paths))
      for idx, key_path in enumerate(init_model_paths):
        im_key, model_path = key_path
        with open(model_path, 'rb') as f:
          model = pickle.load(f)
          if not im_key.startswith('None:'):
            key = 'None:' + im_key
          else:
            key = im_key
          if hasattr(model, 'key'):
            logger.log('__init__: init_model key {} stored in its model '
                       'has been renamed as {}'.format(model.key, key))
          if hasattr(model, 'model'):
            model = model.model
          hyperparam = None
          # specify init_model's hyperparam if possible
          if 'lrn_id_list' in game_mgr_config:
            hyperparam = self._hyper_mgr._default_hyperparam(
              learner_id=game_mgr_config['lrn_id_list'][idx])
            logger.log('__init__: init model {} has been bound with '
                       'hyperparam {}'.format(key, hyperparam))
          t = time.strftime('%Y%m%d%H%M%S')
          self._model_pool_apis.push_model(model, hyperparam, key, t, t, t)
        f.close()
        logger.log('__init__: done pushing {} to model pool'.format(key))
        self.game_mgr.add_player(p=key, parent_p=None)
        logger.log('__init__: done adding player {} to game mgr'.format(key))
        self.init_model_keys.append(key)
    else:
      logger.log('__init__: init_model is None.')

  def _on_request_learner_task(self, learner_id):
    logger.log("_on_request_learner_task: learner_id:'%s'" % str(learner_id))

    # create learning task indicating what params and hyperparams to use
    if learner_id not in self._learner_task_table.keys():
      # A new learner requests learning task for the first time
      hyperparam = self._hyper_mgr.get_learner_hyperparam(
        learner_id, copy_from_model_key=None
      )
      cur_model_key = None
      if hasattr(hyperparam, 'init_model_key'):
        if hyperparam.init_model_key is not None:
          assert hyperparam.init_model_key in self.init_model_keys, \
            'hyperparam init_model_key {} not in init_model_keys {}'.format(
              hyperparam.init_model_key, self.init_model_keys)
        cur_model_key = hyperparam.init_model_key
      elif len(self.init_model_keys) > 0:
        cur_model_key = self.init_model_keys[0]
      model_key = self._gen_new_model_key(old_model_key=cur_model_key)
      task = LearnerTask(model_key, hyperparam,
                         parent_model_key=cur_model_key)
      logger.log("_on_request_learner_task: new learner_id, with model key "
                 "{} and parent model key {}".format(model_key, cur_model_key))
    else:
      # A known existing learner requests learning task
      cur_model_key = self._learner_task_table[learner_id].model_key
      assert cur_model_key is not None
      parent_model_key, is_mutate = self.game_mgr.get_player(cur_model_key)
      if parent_model_key is None:
        # None means continue training, do NOT mutate
        parent_model_key = cur_model_key
        model_key = cur_model_key
        logger.log("_on_request_learner_task:",
                   "continue training with current model key,",
                   "do Not mutate")
      else:
        # continue training with a new model key or new model key
        model_key = self._gen_new_model_key(old_model_key=parent_model_key)
        logger.log("_on_request_learner_task:", "new model key",
                   "mutate {}: ".format(is_mutate))
      hyperparam = self._hyper_mgr.get_learner_hyperparam(
        learner_id, copy_from_model_key=parent_model_key, is_mutate=is_mutate
      )
      task = LearnerTask(model_key, hyperparam,
                         parent_model_key=parent_model_key)
      logger.log("_on_request_learner_task: known learner_id ")
    logger.log("_on_request_learner_task: task: %s" % str(task))

    # view new model_key as new player name and add it to game_mgr
    self.game_mgr.add_player(task.model_key,
                             parent_p=task.parent_model_key,
                             learner_id=learner_id)
    logger.log(
      "_on_request_learner_task: ",
      "done adding player {}, parent_player {}, to game mgr".format(
        task.model_key, task.parent_model_key)
    )

    # book-keep the learner task (either add or update)
    self._learner_task_table[learner_id] = task
    return task

  def _on_query_learner_task(self, learner_id):
    logger.log("_on_query_learner_task: learner_id:'%s'" % str(learner_id))

    if learner_id in self._learner_task_table:
      task = self._learner_task_table[learner_id]
      logger.log('_on_query_learner_task: task: {}'.format(str(task)))
      return task
    else:
      logger.log('_on_query_learner_task: task not exists.')
      return LeagueMgrErroMsg("Learner task not exits.")

  def _on_request_actor_task(self, actor_id, learner_id):
    actor_task = super(LeagueMgr, self)._on_request_actor_task(
        actor_id, learner_id)
    if not isinstance(actor_task, LeagueMgrErroMsg):
      assert isinstance(actor_task, ActorTask)
      self.game_mgr.start_match(actor_task.model_key1, actor_task.model_key2,
                                actor_id)

    logger.log("_on_request_actor_task: %s" % str(actor_task),
               level=logger.DEBUG)
    return actor_task

  def _on_request_eval_actor_task(self, actor_id):
    logger.log("get_eval_actor_task: actor_id:{}".format(str(actor_id)),
               level=logger.DEBUG)
    if len(self.game_mgr.players) > 1:
      rp, cp = self.game_mgr.get_eval_match()
      return ActorTask(rp, cp, None)
    else:
      return LeagueMgrErroMsg("Actor task not ready.")

  def _on_request_train_actor_task(self, actor_id, learner_id):
    logger.log(
      "_on_request_train_actor_task: actor_id:{}, learner_id:{}".format(
        actor_id, learner_id), level=logger.DEBUG
    )
    if (learner_id in self._learner_task_table
        and self._learner_task_table[learner_id].model_key is not None):
      hyperparam = self._learner_task_table[learner_id].hyperparam
      model_key = self._learner_task_table[learner_id].model_key
      oppo_model_key = self.game_mgr.get_opponent(model_key, hyperparam)
      if isinstance(oppo_model_key, LeagueMgrErroMsg):
        logger.log(f'get_opponent not ready: {oppo_model_key}', level=logger.WARN)
        return oppo_model_key
      return ActorTask(model_key, oppo_model_key, hyperparam)
    else:
      if learner_id not in self._learner_task_table:
        logger.log("learner_id({}) hasn't request_learner_task.".format(
          learner_id), level=logger.WARN)
      elif self._learner_task_table[learner_id].model_key is None:
        logger.log("learner_id({}) hasn't notify_learner_task_begin.".format(
          learner_id), level=logger.WARN)
      return LeagueMgrErroMsg("Actor task not ready.")

  def _on_notify_actor_task_end(self, actor_id, match_result):
    # update match result
    m = match_result
    self.game_mgr.finish_match(m.model_key1, m.model_key2, m.outcome, m.info, actor_id)

  def _on_notify_learner_task_begin(self, learner_id, learner_task):
    super(LeagueMgr, self)._on_notify_learner_task_begin(
        learner_id, learner_task)
    assert (learner_task.model_key is not None)

  def _save_checkpoint(self, checkpoint_root, checkpoint_name):
    checkpoint_dir = os.path.join(checkpoint_root, checkpoint_name)
    logger.log('{}saving league-mgr to {}'.format(now(), checkpoint_dir))

    super(LeagueMgr, self)._save_checkpoint(checkpoint_root, checkpoint_name)
    # 1.
    filepath = os.path.join(checkpoint_dir, 'learner_task_table')
    with open(filepath, 'wb') as f:
      pickle.dump(self._learner_task_table, f)
    # 2.
    self._hyper_mgr.save(checkpoint_dir)
    # 3.
    self.game_mgr.save(checkpoint_dir)
    logger.log('{}done saving league-mgr'.format(now()))

  def _restore_checkpoint(self, checkpoint_dir):
    super(LeagueMgr, self)._restore_checkpoint(checkpoint_dir)
    logger.log('{}loading league-mgr from {}'.format(now(), checkpoint_dir))
    # 3.
    self.game_mgr.load(checkpoint_dir)
    # 2.
    self._hyper_mgr.load(checkpoint_dir)
    # 1.
    filepath = os.path.join(checkpoint_dir, 'learner_task_table')
    with open(filepath, 'rb') as f:
      self._learner_task_table = pickle.load(f)
    logger.log('{}done loading league-mgr'.format(now()))

  def _on_request_add_model(self, model):
    self._model_pool_apis.push_model(model.model, model.hyperparam, model.key,
      model.createtime, model.freezetime, model.updatetime)
    self.game_mgr.add_player(model.key)


class PARLeagueMgr(LeagueMgr):
  """ Pseudo Parallel League Manager.

  Conceptually, it maintains pseudo_learner_num learners (e.g., 30), but can run
  over less physical learners (e.g., 8). """

  def __init__(self, port, model_pool_addrs, mutable_hyperparam_type,
               pseudo_learner_num, **kwargs):
    super(PARLeagueMgr, self).__init__(port, model_pool_addrs,
                                       mutable_hyperparam_type, **kwargs)
    self.pseudo_learner_num = pseudo_learner_num
    self.cur_pseudo_id = 0
    self.id_map = {}  # {learner_id: cur_pseudo_id}

  def get_pseudo_id(self, learner_id):
    if learner_id not in self.id_map:
      self.cur_pseudo_id = (self.cur_pseudo_id + 1) % self.pseudo_learner_num
      logger.log("learner {} begins training with pseude learner_id"
                 " {}".format(learner_id, self.cur_pseudo_id))
      self.id_map[learner_id] = self.cur_pseudo_id
    return self.id_map[learner_id]

  def _on_request_learner_task(self, learner_id):
    pseudo_id = self.get_pseudo_id(learner_id)
    return super(PARLeagueMgr, self)._on_request_learner_task(pseudo_id)

  def _on_notify_learner_task_begin(self, learner_id, learner_task):
    pseudo_id = self.get_pseudo_id(learner_id)
    return super(PARLeagueMgr, self)._on_notify_learner_task_begin(pseudo_id,
                                                                   learner_task)

  def _on_notify_learner_task_end(self, learner_id):
    pseudo_id = self.get_pseudo_id(learner_id)
    self.cur_pseudo_id = (self.cur_pseudo_id + 1) % self.pseudo_learner_num
    self.id_map[learner_id] = self.cur_pseudo_id
    logger.log("learner {} switches from pseudo learner_id {} to "
               "pseudo learner_id {}".format(learner_id, pseudo_id,
                                             self.id_map[learner_id]))
    return super(PARLeagueMgr, self)._on_notify_learner_task_end(pseudo_id)

  def _on_request_actor_task(self, actor_id, learner_id):
    pseude_id = (self.id_map[learner_id] if learner_id in self.id_map
                 else learner_id)
    return super(PARLeagueMgr, self)._on_request_actor_task(actor_id, pseude_id)
