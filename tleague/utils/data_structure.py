import numpy as np
from tensorflow.contrib.framework import nest
from tpolicies.tp_utils import map_gym_space_to_structure, template_structure_from_gym_space
from tpolicies.utils.distributions import make_pdtype


def namedlist(fields):
  if isinstance(fields, str):
    fields = fields.replace(',', ' ').split()
  fields = list(map(str, fields))
  class Structure(list):
    _fields = fields
    _attr_index = {field: i for i, field in enumerate(fields)}
    def __init__(self, seq=()):
      if len(seq) > len(fields):
        raise IOError('input is too long!')
      else:
        seq = list(seq) + [None] * (len(fields) - len(seq))
      super(Structure, self).__init__(seq)
    def __getattr__(self, item):
      if item not in self._attr_index:
        raise AttributeError(f'namedlist object has no attribute {item}')
      return self[self._attr_index[item]]
    def __setattr__(self, key, value):
      if key not in self._attr_index:
        raise AttributeError(f'namedlist object has no attribute {key}')
      self[self._attr_index[key]] = value
  return Structure


class DataStructure(object):
  def __init__(self, fields, specs, templates):
    self.fields = fields # dict fields
    self._structure = namedlist(fields)
    self.spec = self.structure(specs) # whole data structure
    self.template_spec = self.structure(templates)
    self.flatten_spec = nest.flatten_up_to(self.template_spec, self.spec)

  def flatten(self, struct_input):
    return tuple(nest.flatten_up_to(self.template_spec, struct_input))

  def make_structure(self, flat_input):
    return nest.pack_sequence_as(self.template_spec, flat_input)

  def structure(self, data):
    return self._structure(data)


class ILData(DataStructure):
  def __init__(self, ob_space, ac_space, use_lstm=False, hs_len=None):
    shape_dtype = lambda x: (x.shape, x.dtype)
    _fields = ['X', 'A']
    specs = [map_gym_space_to_structure(shape_dtype, ob_space),
             map_gym_space_to_structure(shape_dtype, ac_space),]
    templates = [template_structure_from_gym_space(ob_space),
                 template_structure_from_gym_space(ac_space),]
    if use_lstm:
      assert int(hs_len) == hs_len
      _fields.extend(['S', 'M'])
      specs.extend([([hs_len], np.float32),
                    ([], np.bool),])
      templates.extend([None, None,])
    super(ILData, self).__init__(_fields, specs, templates)


class PGData(DataStructure):
  def __init__(self, ob_space, ac_space, n_v, use_lstm=False, hs_len=None,
               distillation=False, use_oppo_data=False):
    _fields = ['X', 'A', 'neglogp']
    shape_dtype = lambda x: (x.shape, x.dtype)
    logit_shape_dtype = lambda x: (make_pdtype(x).param_shape(), np.float32)
    neglogp_shape_dtype = map_gym_space_to_structure(lambda x: ([], np.float32), ac_space)
    neglogp_templates = template_structure_from_gym_space(ac_space)
    logits_shape_dtype = map_gym_space_to_structure(logit_shape_dtype, ac_space)
    logits_templates = template_structure_from_gym_space(ac_space)
    specs = [map_gym_space_to_structure(shape_dtype, ob_space),
             map_gym_space_to_structure(shape_dtype, ac_space),
             neglogp_shape_dtype]
    templates = [template_structure_from_gym_space(ob_space),
                 template_structure_from_gym_space(ac_space),
                 neglogp_templates]
    if use_lstm:
      assert int(hs_len) == hs_len
      _fields.extend(['S', 'M'])
      specs.extend([([hs_len], np.float32),
                    ([], np.bool), ])
      templates.extend([None, None, ])
    if distillation:
      _fields.append('logits')
      specs.append(logits_shape_dtype)
      templates.append(logits_templates)
    if use_oppo_data:
      _fields.append('OPPO_X')
      specs.append(map_gym_space_to_structure(shape_dtype, ob_space))
      templates.append(template_structure_from_gym_space(ob_space))
      if use_lstm:
        _fields.append('OPPO_S')  # oppo's mask is the same as self
        specs.append(([hs_len], np.float32))
        templates.append(None)
    self.specs = specs
    self.templates = templates
    super(PGData, self).__init__(_fields, specs, templates)


class PPOData(PGData):
  # old data structure. logp & logit are long vectors
  def __init__(self, ob_space, ac_space, n_v, use_lstm=False, hs_len=None,
               distillation=False):
    super(PPOData, self).__init__(ob_space, ac_space, n_v, use_lstm,
                                  hs_len, distillation)
    self.fields.extend(['R', 'V'])
    self.specs.extend([([n_v], np.float32),
                       ([n_v], np.float32)])
    self.templates.extend([None, None])
    super(PGData, self).__init__(self.fields, self.specs, self.templates)


class VtraceData(PGData):
  def __init__(self, ob_space, ac_space, n_v, use_lstm=False, hs_len=None,
               distillation=False):
    super(VtraceData, self).__init__(ob_space, ac_space, n_v, use_lstm,
                                     hs_len, distillation)
    self.fields.extend(['r', 'discount'])
    self.specs.extend([([n_v], np.float32),
                       ([], np.float32)])
    self.templates.extend([None, None])
    super(PGData, self).__init__(self.fields, self.specs, self.templates)


class InfData(DataStructure):
  def __init__(self, ob_space, ac_space, use_self_fed_heads=True,
               use_lstm=False, hs_len=None):
    shape_dtype = lambda x: (x.shape, x.dtype)
    _fields = ['X']
    specs = [map_gym_space_to_structure(shape_dtype, ob_space)]
    templates = [template_structure_from_gym_space(ob_space)]
    if not use_self_fed_heads:
      _fields.append('A')
      specs.append(map_gym_space_to_structure(shape_dtype, ac_space))
      templates.append(template_structure_from_gym_space(ac_space))
    if use_lstm:
      assert int(hs_len) == hs_len
      _fields.extend(['S', 'M'])
      specs.extend([([hs_len], np.float32),
                    ([], np.bool),])
      templates.extend([None, None,])
    super(InfData, self).__init__(_fields, specs, templates)
