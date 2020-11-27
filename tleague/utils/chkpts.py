import os
import pickle
import time

from tleague.utils import logger
from tleague.utils import now
from tleague.model_pools.model import Model


class ChkptsFromModelPool(object):
  def __init__(self, model_pool_apis, save_learner_meta=False):
    self._model_pool_apis = model_pool_apis
    self._save_learner_meta = save_learner_meta

  def _save_model_checkpoint(self, checkpoint_root, checkpoint_name):
    checkpoint_dir = os.path.join(checkpoint_root, checkpoint_name)
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    logger.log(now() + 'Pulling updatetime')
    updatetime_dict = self._model_pool_apis.pull_all_attr('updatetime')
    logger.log(now() + 'Done pulling updatetime, no.={}'.format(
      len(updatetime_dict)
    ))

    filenames = []
    for model_key, updatetime in updatetime_dict.items():
      filename = "%s_%s.model" % (model_key, updatetime)
      filepath = os.path.join(checkpoint_root, filename)
      filenames.append(filename + '\n')
      if not os.path.isfile(filepath):
        logger.log(now() + 'Pulling model {}'.format(model_key))
        model = self._model_pool_apis.pull_model(model_key)
        logger.log(now() + 'Done pulling model {}'.format(model_key))
        assert model_key == model.key
        with open(filepath, 'wb') as f:
          pickle.dump(model, f)
          if self._save_learner_meta:
            learner_meta = self._model_pool_apis.pull_learner_meta(model_key)
            pickle.dump(learner_meta, f)
          logger.log(now() + 'Saved model to {}'.format(f.name))
    filelistpath = os.path.join(checkpoint_dir, 'filename.list')
    with open(filelistpath, 'w') as f:
      f.writelines(filenames)
    with open(os.path.join(checkpoint_dir, '.ready'), 'w') as f:
      f.write('ready')
      f.flush()

  def _restore_model_checkpoint(self, checkpoint_dir):
    ''' Assume all the models are stored in the parent folder of checkpoint_dir '''
    filelistpath = os.path.join(checkpoint_dir, 'filename.list')
    modelpath = os.path.split(os.path.abspath(checkpoint_dir))[0] # parent folder of checkpoint_dir
    logger.log('Read model file names from %s.' % filelistpath)
    with open(filelistpath, 'r') as f:
      filenames = f.readlines()
    logger.log('Restore models from %s' % modelpath)
    learner_meta = None
    max_key_idx = 0
    for filename in filenames:
      filename = filename.strip()
      filepath = os.path.join(modelpath, filename)
      if not os.path.isfile(filepath):
        print(filename + 'is missing!')
      with open(filepath, 'rb') as f:
        model = pickle.load(f)
        if self._save_learner_meta:
          learner_meta = pickle.load(f)
      key = filename.split('.')[0][:-15]
      assert key == model.key
      self._model_pool_apis.push_model(model.model, model.hyperparam, model.key,
                                       model.createtime, model.freezetime,
                                       model.updatetime, learner_meta)
      time.sleep(0.1)
      if key.split(':')[-1].isdigit():
        max_key_idx = max(max_key_idx, int(key.split(':')[-1]))
    logger.log('Restore models finished.')
    return max_key_idx


class ChkptsFromSelf(object):
  def __init__(self, read_op, load_op, model_key):
    self._read_op = read_op
    self._load_op = load_op
    self._model_key = model_key

  def _save_model_checkpoint(self, checkpoint_root, checkpoint_name=None):
    updatetime = time.strftime('%Y%m%d%H%M%S')
    if checkpoint_name is None:
      checkpoint_name = "checkpoint_%s" % updatetime
    checkpoint_dir = os.path.join(checkpoint_root, checkpoint_name)
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    filename = "%s_%s.model" % (self._model_key, updatetime)
    filepath = os.path.join(checkpoint_root, filename)
    model = self._read_op()
    with open(filepath, 'wb') as f:
      pickle.dump(Model(model, None, self._model_key,
                        None, None, updatetime), f)
    filelistpath = os.path.join(checkpoint_dir, 'filename.list')
    with open(filelistpath, 'w') as f:
      f.write(filename)
    with open(os.path.join(checkpoint_dir, '.ready'), 'w') as f:
      f.write('ready')
      f.flush()

  def _restore_model_checkpoint(self, checkpoint_dir):
    ''' Assume all the models are stored in the parent folder of checkpoint_dir '''
    filelistpath = os.path.join(checkpoint_dir, 'filename.list')
    modelpath = os.path.split(os.path.abspath(checkpoint_dir))[0] # parent folder of checkpoint_dir
    logger.log('Read model file names from %s.' % filelistpath)
    with open(filelistpath, 'r') as f:
      filename = f.read().strip()
    logger.log('Restore models from %s' % modelpath)
    filepath = os.path.join(modelpath, filename)
    if not os.path.isfile(filepath):
      raise EOFError(filename + 'not exists!')
    with open(filepath, 'rb') as f:
      model = pickle.load(f)
    self._load_op(model.model)
    logger.log('Restore models finished.')


def test1():
  from tleague.model_pools.model_pool import ModelPool
  from tleague.model_pools.model_pool_apis import ModelPoolAPIs
  from multiprocessing import Process

  server_process = Process(target=lambda: ModelPool(ports="11001:11006").run())
  server_process.start()

  model_pool_apis = ModelPoolAPIs(["localhost:11001:11006"])
  model_pool_apis.push_model('model1', None, 'model1')

  saver = ChkptsFromModelPool(model_pool_apis)
  saver._save_model_checkpoint('./', 'test')
  model_pool_apis.push_model('Modified_model1', None, 'model1')

  saver._restore_model_checkpoint('./test')
  model = model_pool_apis.pull_model('model1')
  server_process.terminate()


def test2():
  def read():
    return 'model1'

  def load(loaded_params):
    print(loaded_params)

  saver = ChkptsFromSelf(read, load, 'ILmodel')
  saver._save_model_checkpoint('./', 'test')
  saver._restore_model_checkpoint('./test')
