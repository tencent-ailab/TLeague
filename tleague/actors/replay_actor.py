from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
from queue import Queue
from threading import Thread
import platform
import random

import numpy as np
from pysc2 import run_configs
from pysc2.lib import point
from s2clientprotocol import sc2api_pb2 as sc_pb
from timitate.utils.utils import map_name_transform
from timitate.utils.const import MAP_ORI_SIZE_DICT, MAP_PLAYABLE_SIZE_DICT
from timitate.utils import pb2pb

from tleague.utils import logger
from tleague.utils.io import TensorZipper
from tleague.utils.data_structure import ILData
from tleague.learners.learner_apis import ImLearnerAPIs
from tleague.model_pools.model_pool_apis import ModelPoolAPIs
from tleague.actors.agent import PGAgent


def _get_interface(map_name, game_core_config={}):
  crop = (game_core_config['crop_to_playable_area']
          if 'crop_to_playable_area' in game_core_config else False)
  map_name = map_name_transform(map_name)
  map_size = MAP_PLAYABLE_SIZE_DICT[map_name] if crop else MAP_ORI_SIZE_DICT[
    map_name]
  screen_size = point.Point(16, 16)
  minimap_size = point.Point(int(map_size[0]), int(map_size[1]))
  interface = sc_pb.InterfaceOptions(
    raw=True, score=True, feature_layer=sc_pb.SpatialCameraSetup(width=24))
  screen_size.assign_to(interface.feature_layer.resolution)
  minimap_size.assign_to(interface.feature_layer.minimap_resolution)
  if crop:
    interface.feature_layer.crop_to_playable_area = True
    interface.raw_crop_to_playable_area = True
  if 'show_cloaked' in game_core_config:
    interface.show_cloaked = game_core_config['show_cloaked']
  if 'show_burrowed_shadows' in game_core_config:
    interface.show_burrowed_shadows = game_core_config['show_burrowed_shadows']
  if 'show_placeholders' in game_core_config:
    interface.show_placeholders = game_core_config['show_placeholders']
  return interface


class ReplayExtractor(object):

  def __init__(self, replay_dir, replay_filename, player_id, replay_converter,
               step_mul=8, version=None, da_rate=-1.,
               game_core_config={}, unk_mmr_dft_to=4000):
    self._replay_filepath = os.path.join(replay_dir, replay_filename)
    self._replay_name = replay_filename.split('.')[0]
    self._player_id = player_id
    self._step_mul = step_mul
    self._replay_converter = replay_converter
    self._version = version
    self._da_rate = da_rate
    self._game_core_config = game_core_config
    self._unk_mmr_dft_to = unk_mmr_dft_to

  def extract(self):
    try:
      for frame in self._extract():
        yield frame
    except Exception as e:
      logger.log("Extract replay[%s] player[%d] failed: %s" % (
        self._replay_filepath, self._player_id, e), level=logger.WARN)
      raise e

  def _extract(self):
    run_config = run_configs.get()
    replay_data = run_config.replay_data(self._replay_filepath)
    with run_config.start(version=self._version) as controller:
      replay_info = controller.replay_info(replay_data)
      mmr = None
      for p in replay_info.player_info:
        if p.player_info.player_id == self._player_id:
          mmr = p.player_mmr
      assert mmr is not None
      if mmr < 1:
        logger.log("Encounter unknown mmr: {}, defaults it to: {}".format(
          mmr, self._unk_mmr_dft_to
        ))
        mmr = self._unk_mmr_dft_to
      #map_name = 'KairosJunction'
      map_name = replay_info.map_name
     
      # set map_data when available
      map_data = None
      if replay_info.local_map_path:
        map_data = run_config.map_data(replay_info.local_map_path)
        print('using local_map_path {}'.format(replay_info.local_map_path))
      controller.start_replay(sc_pb.RequestStartReplay(
        replay_data=replay_data,
        map_data=map_data,
        options=_get_interface(map_name, game_core_config=self._game_core_config),
        observed_player_id=self._player_id,
        disable_fog=False))
      controller.step()
      start_pos = None
      enable_da = False
      if self._da_rate > 0:
        if random.random() < self._da_rate:
          enable_da = True
          logger.log("data augmentation: revise")
          game_info = controller.game_info()
          start_pos = game_info.start_raw.start_locations[0]
          pb2pb.replace_loc(start_pos, map_name)
        else:
          logger.log("data augmentation: not revise")
          
      self._replay_converter.reset(
        replay_name=self._replay_name,
        player_id=self._player_id,
        mmr=mmr,
        map_name=map_name,
        start_pos=start_pos)

      last_obs, last_game_info = None, None
      while True:
        obs = controller.observe()
        if enable_da:
          obs = pb2pb.make_aug_data(obs, map_name)
        if obs.player_result:
          samples = self._replay_converter.convert(pb=(obs, None), next_pb=None)
          for data in samples:
            yield data
          break
        # game_info = controller.game_info()
        game_info = None
        if last_obs is not None:
          samples = self._replay_converter.convert(
            pb=(last_obs, last_game_info),
            next_pb=(obs, game_info))
          for data in samples:
            yield data
        last_obs, last_game_info = obs, game_info
        controller.step(self._step_mul)


class ReplayActor(object):
  def __init__(self, learner_addr, replay_dir, replay_converter_type,
               policy=None, policy_config=None, model_pool_addrs=None,
               n_v=1, log_interval=50, step_mul=8, SC2_bin_root='/root/',
               game_version='3.16.1', unroll_length=32, update_model_freq=32,
               converter_config=None, agent_cls=None, infserver_addr=None,
               compress=True, da_rate=-1., unk_mmr_dft_to=4000,
               post_process_data=None):
    self._data_pool_apis = ImLearnerAPIs(learner_addr)
    self._SC2_bin_root = SC2_bin_root
    self._log_interval = log_interval
    self._replay_dir = replay_dir
    self._step_mul = step_mul
    self._game_version = game_version
    self._unroll_length = unroll_length
    self._data_queue = Queue(unroll_length)
    self._push_thread = Thread(target=self._push_data, args=(self._data_queue,))
    self._push_thread.daemon = True
    self._push_thread.start()
    self.converter_config = {} if converter_config is None else converter_config
    self.converter_config['game_version'] = game_version
    self.replay_converter_type = replay_converter_type
    self._replay_converter = replay_converter_type(**self.converter_config)
    self._use_policy = policy is not None
    self._update_model_freq = update_model_freq
    self.model_key = 'IL-model'
    self._da_rate = da_rate
    self._unk_mmr_dft_to = unk_mmr_dft_to
    self._system = platform.system()
    self._post_process_data = post_process_data
    ob_space, ac_space = self._replay_converter.space
    if self._post_process_data:
      ob_space, ac_space = self._post_process_data(ob_space, ac_space)
    if self._use_policy:
      self.model = None
      policy_config = {} if policy_config is None else policy_config
      agent_cls = agent_cls or PGAgent
      policy_config['batch_size'] = 1
      policy_config['rollout_len'] = 1
      policy_config['use_loss_type'] = 'none'
      self.infserver_addr = infserver_addr
      self.agent = agent_cls(policy, ob_space, ac_space, n_v=n_v,
                             scope_name="self", policy_config=policy_config,
                             use_gpu_id=-1, infserver_addr=infserver_addr,
                             compress=compress)
      if infserver_addr is None:
        self._model_pool_apis = ModelPoolAPIs(model_pool_addrs)
    self.ds = ILData(ob_space, ac_space, self._use_policy, 1) # hs_len does not matter

  def run(self):
    self.replay_task = self._data_pool_apis.request_replay_task()
    while self.replay_task != "":
      game_version = self.replay_task.game_version or self._game_version
      self._adapt_system(game_version)
      if game_version != self._game_version:
        # need re-init replay converter
        self._game_version = game_version
        self.converter_config['game_version'] = game_version
        self._replay_converter = self.replay_converter_type(**self.converter_config)
      game_core_config = (
        {} if 'game_core_config' not in self.converter_config
        else self.converter_config['game_core_config']
      )
      extractor = ReplayExtractor(
        replay_dir=self._replay_dir,
        replay_filename=self.replay_task.replay_name,
        player_id=self.replay_task.player_id,
        replay_converter=self._replay_converter,
        step_mul=self._step_mul,
        version=game_version,
        game_core_config=game_core_config,
        da_rate=self._da_rate,
        unk_mmr_dft_to=self._unk_mmr_dft_to
      )
      self._steps = 0
      first_frame = True
      if self._use_policy:
        self.agent.reset()
        self._update_agent_model()
      for frame in extractor.extract():
        if self._post_process_data:
          obs, act = self._post_process_data(*frame[0])
        else:
          obs, act = frame[0]
        if self._use_policy:
          data = (obs, act, self.agent.state, np.array(first_frame, np.bool))
          self.agent.update_state(obs)
          first_frame = False
        else:
          data = (obs, act)
        data = self.ds.flatten(self.ds.structure(data))
        if self._data_queue.full():
          logger.log("Actor's queue is full.", level=logger.WARN)
        self._data_queue.put((TensorZipper.compress(data), frame[1]))
        logger.log('successfully put one tuple.', level=logger.DEBUG)
        self._steps += 1
        if self._steps % self._log_interval == 0:
          logger.log("%d frames of replay task [%s] sent to learner." % (
            self._steps, self.replay_task))
        if self._use_policy and self._steps % self._update_model_freq == 0:
          self._update_agent_model()
      logger.log("Replay task [%s] done. %d frames sent to learner." % (
        self.replay_task, self._steps))
      self.replay_task = self._data_pool_apis.request_replay_task()
    logger.log("All tasks done.")

  def _adapt_system(self, game_version):
    # TODO(pengsun): any stuff for Darwin, Window?
    if self._system == 'Linux':
      # set the SC2PATH for sc2 binary. See deepmind/pysc2 doc.
      if game_version != '4.7.1' or 'SC2PATH' in os.environ:
       os.environ['SC2PATH']=os.path.join(self._SC2_bin_root, game_version)
    return

  def _update_agent_model(self):
    if self.infserver_addr is not None:
      return
    logger.log('entering _update_agents_model', 'steps: {}'.format(self._steps),
               level=logger.DEBUG + 5)
    if self._should_update_model(self.model, self.model_key):
      model = self._model_pool_apis.pull_model(self.model_key)
      self.agent.load_model(model.model)
      self.model = model

  def _should_update_model(self, model, model_key):
    if model is None:
      return True
    else:
      return self._model_pool_apis.pull_attr('updatetime',
                                             model_key) > model.updatetime

  def _push_data(self, data_queue):
    """ push trajectory for the learning agent (id 0). Invoked in a thread """
    while data_queue.empty():
      time.sleep(5)
    logger.log('entering _push_data_to_learner', 'steps: {}'.format(self._steps),
               level=logger.DEBUG + 5)
    while True:
      task = self.replay_task
      frames = []
      weights = []
      for _ in range(self._unroll_length):
        frame, weight = data_queue.get()
        frames.append(frame)
        weights.append(weight)
      self._data_pool_apis.push_data((task, frames, weights))
