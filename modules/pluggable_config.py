# -*- coding: utf-8 -*-
from operator import attrgetter
from pyangbind.lib.yangtypes import RestrictedPrecisionDecimalType
from pyangbind.lib.yangtypes import RestrictedClassType
from pyangbind.lib.yangtypes import TypedListType
from pyangbind.lib.yangtypes import YANGBool
from pyangbind.lib.yangtypes import YANGListType
from pyangbind.lib.yangtypes import YANGDynClass
from pyangbind.lib.yangtypes import ReferenceType
from pyangbind.lib.yangtypes import YANGBinary
from pyangbind.lib.yangtypes import YANGBitsType
from pyangbind.lib.base import PybindBase
from collections import OrderedDict
from decimal import Decimal

import builtins as __builtin__
long = int
class yc_pluggable_settings_pluggable_config__pluggables_pluggable_pluggable_settings(PybindBase):
  """
  This class was auto-generated by the PythonClass plugin for PYANG
  from YANG module pluggable-config - based on the path /pluggables/pluggable/pluggable-settings. Each member element of
  the container is represented as a class variable - with a specific
  YANG type.

  YANG Description: Configuration and operational state of the pluggable.
  """
  __slots__ = ('_path_helper', '_extmethods', '__frequency','__power','__config_time',)

  _yang_name = 'pluggable-settings'
  _yang_namespace = 'urn:pluggable-config:1.0'

  _pybind_generated_by = 'container'

  def __init__(self, *args, **kwargs):

    self._path_helper = False

    self._extmethods = False
    self.__frequency = YANGDynClass(base=RestrictedClassType(base_type=RestrictedClassType(base_type=long, restriction_dict={'range': ['0..4294967295']}, int_size=32), restriction_dict={'range': ['191300..196100']}), is_leaf=True, yang_name="frequency", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='uint32', is_config=True)
    self.__power = YANGDynClass(base=RestrictedClassType(base_type=Decimal, restriction_dict={'range': ['-6.0..1.0']}), default=Decimal(0), is_leaf=True, yang_name="power", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='decimal64', is_config=True)
    self.__config_time = YANGDynClass(base=RestrictedClassType(base_type=Decimal, restriction_dict={'range': ['0.0..150.0']}), is_leaf=True, yang_name="config_time", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='decimal64', is_config=True)

    load = kwargs.pop("load", None)
    if args:
      if len(args) > 1:
        raise TypeError("cannot create a YANG container with >1 argument")
      all_attr = True
      for e in self._pyangbind_elements:
        if not hasattr(args[0], e):
          all_attr = False
          break
      if not all_attr:
        raise ValueError("Supplied object did not have the correct attributes")
      for e in self._pyangbind_elements:
        nobj = getattr(args[0], e)
        if nobj._changed() is False:
          continue
        setmethod = getattr(self, "_set_%s" % e)
        if load is None:
          setmethod(getattr(args[0], e))
        else:
          setmethod(getattr(args[0], e), load=load)

  def _path(self):
    if hasattr(self, "_parent"):
      return self._parent._path()+[self._yang_name]
    else:
      return ['pluggables', 'pluggable', 'pluggable-settings']

  def _get_frequency(self):
    """
    Getter method for frequency, mapped from YANG variable /pluggables/pluggable/pluggable_settings/frequency (uint32)

    YANG Description: The frequency of the device in GHz.
    """
    return self.__frequency
      
  def _set_frequency(self, v, load=False):
    """
    Setter method for frequency, mapped from YANG variable /pluggables/pluggable/pluggable_settings/frequency (uint32)
    If this variable is read-only (config: false) in the
    source YANG file, then _set_frequency is considered as a private
    method. Backends looking to populate this variable should
    do so via calling thisObj._set_frequency() directly.

    YANG Description: The frequency of the device in GHz.
    """
    if hasattr(v, "_utype"):
      v = v._utype(v)
    try:
      t = YANGDynClass(v,base=RestrictedClassType(base_type=RestrictedClassType(base_type=long, restriction_dict={'range': ['0..4294967295']}, int_size=32), restriction_dict={'range': ['191300..196100']}), is_leaf=True, yang_name="frequency", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='uint32', is_config=True)
    except (TypeError, ValueError):
      raise ValueError({
          'error-string': """frequency must be of a type compatible with uint32""",
          'defined-type': "uint32",
          'generated-type': """YANGDynClass(base=RestrictedClassType(base_type=RestrictedClassType(base_type=long, restriction_dict={'range': ['0..4294967295']}, int_size=32), restriction_dict={'range': ['191300..196100']}), is_leaf=True, yang_name="frequency", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='uint32', is_config=True)""",
        })

    self.__frequency = t
    if hasattr(self, '_set'):
      self._set()

  def _unset_frequency(self):
    self.__frequency = YANGDynClass(base=RestrictedClassType(base_type=RestrictedClassType(base_type=long, restriction_dict={'range': ['0..4294967295']}, int_size=32), restriction_dict={'range': ['191300..196100']}), is_leaf=True, yang_name="frequency", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='uint32', is_config=True)


  def _get_power(self):
    """
    Getter method for power, mapped from YANG variable /pluggables/pluggable/pluggable_settings/power (decimal64)

    YANG Description: The power of the pluggable in mWatts per dB.
    """
    return self.__power
      
  def _set_power(self, v, load=False):
    """
    Setter method for power, mapped from YANG variable /pluggables/pluggable/pluggable_settings/power (decimal64)
    If this variable is read-only (config: false) in the
    source YANG file, then _set_power is considered as a private
    method. Backends looking to populate this variable should
    do so via calling thisObj._set_power() directly.

    YANG Description: The power of the pluggable in mWatts per dB.
    """
    if hasattr(v, "_utype"):
      v = v._utype(v)
    try:
      t = YANGDynClass(v,base=RestrictedClassType(base_type=Decimal, restriction_dict={'range': ['-6.0..1.0']}), default=Decimal(0), is_leaf=True, yang_name="power", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='decimal64', is_config=True)
    except (TypeError, ValueError):
      raise ValueError({
          'error-string': """power must be of a type compatible with decimal64""",
          'defined-type': "decimal64",
          'generated-type': """YANGDynClass(base=RestrictedClassType(base_type=Decimal, restriction_dict={'range': ['-6.0..1.0']}), default=Decimal(0), is_leaf=True, yang_name="power", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='decimal64', is_config=True)""",
        })

    self.__power = t
    if hasattr(self, '_set'):
      self._set()

  def _unset_power(self):
    self.__power = YANGDynClass(base=RestrictedClassType(base_type=Decimal, restriction_dict={'range': ['-6.0..1.0']}), default=Decimal(0), is_leaf=True, yang_name="power", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='decimal64', is_config=True)


  def _get_config_time(self):
    """
    Getter method for config_time, mapped from YANG variable /pluggables/pluggable/pluggable_settings/config_time (decimal64)
    """
    return self.__config_time
      
  def _set_config_time(self, v, load=False):
    """
    Setter method for config_time, mapped from YANG variable /pluggables/pluggable/pluggable_settings/config_time (decimal64)
    If this variable is read-only (config: false) in the
    source YANG file, then _set_config_time is considered as a private
    method. Backends looking to populate this variable should
    do so via calling thisObj._set_config_time() directly.
    """
    if hasattr(v, "_utype"):
      v = v._utype(v)
    try:
      t = YANGDynClass(v,base=RestrictedClassType(base_type=Decimal, restriction_dict={'range': ['0.0..150.0']}), is_leaf=True, yang_name="config_time", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='decimal64', is_config=True)
    except (TypeError, ValueError):
      raise ValueError({
          'error-string': """config_time must be of a type compatible with decimal64""",
          'defined-type': "decimal64",
          'generated-type': """YANGDynClass(base=RestrictedClassType(base_type=Decimal, restriction_dict={'range': ['0.0..150.0']}), is_leaf=True, yang_name="config_time", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='decimal64', is_config=True)""",
        })

    self.__config_time = t
    if hasattr(self, '_set'):
      self._set()

  def _unset_config_time(self):
    self.__config_time = YANGDynClass(base=RestrictedClassType(base_type=Decimal, restriction_dict={'range': ['0.0..150.0']}), is_leaf=True, yang_name="config_time", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='decimal64', is_config=True)

  frequency = __builtin__.property(_get_frequency, _set_frequency)
  power = __builtin__.property(_get_power, _set_power)
  config_time = __builtin__.property(_get_config_time, _set_config_time)


  _pyangbind_elements = OrderedDict([('frequency', frequency), ('power', power), ('config_time', config_time), ])


class yc_pluggable_pluggable_config__pluggables_pluggable(PybindBase):
  """
  This class was auto-generated by the PythonClass plugin for PYANG
  from YANG module pluggable-config - based on the path /pluggables/pluggable. Each member element of
  the container is represented as a class variable - with a specific
  YANG type.

  YANG Description: List of pluggables and their configurations.
  """
  __slots__ = ('_path_helper', '_extmethods', '__pluggable_id','__pluggable_settings',)

  _yang_name = 'pluggable'
  _yang_namespace = 'urn:pluggable-config:1.0'

  _pybind_generated_by = 'container'

  def __init__(self, *args, **kwargs):

    self._path_helper = False

    self._extmethods = False
    self.__pluggable_id = YANGDynClass(base=str, is_leaf=True, yang_name="pluggable-id", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, is_keyval=True, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='string', is_config=True)
    self.__pluggable_settings = YANGDynClass(base=yc_pluggable_settings_pluggable_config__pluggables_pluggable_pluggable_settings, is_container='container', yang_name="pluggable-settings", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, extensions=None, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='container', is_config=True)

    load = kwargs.pop("load", None)
    if args:
      if len(args) > 1:
        raise TypeError("cannot create a YANG container with >1 argument")
      all_attr = True
      for e in self._pyangbind_elements:
        if not hasattr(args[0], e):
          all_attr = False
          break
      if not all_attr:
        raise ValueError("Supplied object did not have the correct attributes")
      for e in self._pyangbind_elements:
        nobj = getattr(args[0], e)
        if nobj._changed() is False:
          continue
        setmethod = getattr(self, "_set_%s" % e)
        if load is None:
          setmethod(getattr(args[0], e))
        else:
          setmethod(getattr(args[0], e), load=load)

  def _path(self):
    if hasattr(self, "_parent"):
      return self._parent._path()+[self._yang_name]
    else:
      return ['pluggables', 'pluggable']

  def _get_pluggable_id(self):
    """
    Getter method for pluggable_id, mapped from YANG variable /pluggables/pluggable/pluggable_id (string)

    YANG Description: Unique identifier for the pluggable.
    """
    return self.__pluggable_id
      
  def _set_pluggable_id(self, v, load=False):
    """
    Setter method for pluggable_id, mapped from YANG variable /pluggables/pluggable/pluggable_id (string)
    If this variable is read-only (config: false) in the
    source YANG file, then _set_pluggable_id is considered as a private
    method. Backends looking to populate this variable should
    do so via calling thisObj._set_pluggable_id() directly.

    YANG Description: Unique identifier for the pluggable.
    """
    parent = getattr(self, "_parent", None)
    if parent is not None and load is False:
      raise AttributeError("Cannot set keys directly when" +
                             " within an instantiated list")

    if hasattr(v, "_utype"):
      v = v._utype(v)
    try:
      t = YANGDynClass(v,base=str, is_leaf=True, yang_name="pluggable-id", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, is_keyval=True, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='string', is_config=True)
    except (TypeError, ValueError):
      raise ValueError({
          'error-string': """pluggable_id must be of a type compatible with string""",
          'defined-type': "string",
          'generated-type': """YANGDynClass(base=str, is_leaf=True, yang_name="pluggable-id", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, is_keyval=True, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='string', is_config=True)""",
        })

    self.__pluggable_id = t
    if hasattr(self, '_set'):
      self._set()

  def _unset_pluggable_id(self):
    self.__pluggable_id = YANGDynClass(base=str, is_leaf=True, yang_name="pluggable-id", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, is_keyval=True, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='string', is_config=True)


  def _get_pluggable_settings(self):
    """
    Getter method for pluggable_settings, mapped from YANG variable /pluggables/pluggable/pluggable_settings (container)

    YANG Description: Configuration and operational state of the pluggable.
    """
    return self.__pluggable_settings
      
  def _set_pluggable_settings(self, v, load=False):
    """
    Setter method for pluggable_settings, mapped from YANG variable /pluggables/pluggable/pluggable_settings (container)
    If this variable is read-only (config: false) in the
    source YANG file, then _set_pluggable_settings is considered as a private
    method. Backends looking to populate this variable should
    do so via calling thisObj._set_pluggable_settings() directly.

    YANG Description: Configuration and operational state of the pluggable.
    """
    if hasattr(v, "_utype"):
      v = v._utype(v)
    try:
      t = YANGDynClass(v,base=yc_pluggable_settings_pluggable_config__pluggables_pluggable_pluggable_settings, is_container='container', yang_name="pluggable-settings", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, extensions=None, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='container', is_config=True)
    except (TypeError, ValueError):
      raise ValueError({
          'error-string': """pluggable_settings must be of a type compatible with container""",
          'defined-type': "container",
          'generated-type': """YANGDynClass(base=yc_pluggable_settings_pluggable_config__pluggables_pluggable_pluggable_settings, is_container='container', yang_name="pluggable-settings", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, extensions=None, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='container', is_config=True)""",
        })

    self.__pluggable_settings = t
    if hasattr(self, '_set'):
      self._set()

  def _unset_pluggable_settings(self):
    self.__pluggable_settings = YANGDynClass(base=yc_pluggable_settings_pluggable_config__pluggables_pluggable_pluggable_settings, is_container='container', yang_name="pluggable-settings", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, extensions=None, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='container', is_config=True)

  pluggable_id = __builtin__.property(_get_pluggable_id, _set_pluggable_id)
  pluggable_settings = __builtin__.property(_get_pluggable_settings, _set_pluggable_settings)


  _pyangbind_elements = OrderedDict([('pluggable_id', pluggable_id), ('pluggable_settings', pluggable_settings), ])


class yc_pluggables_pluggable_config__pluggables(PybindBase):
  """
  This class was auto-generated by the PythonClass plugin for PYANG
  from YANG module pluggable-config - based on the path /pluggables. Each member element of
  the container is represented as a class variable - with a specific
  YANG type.

  YANG Description: Container for pluggables.
  """
  __slots__ = ('_path_helper', '_extmethods', '__pluggable',)

  _yang_name = 'pluggables'
  _yang_namespace = 'urn:pluggable-config:1.0'

  _pybind_generated_by = 'container'

  def __init__(self, *args, **kwargs):

    self._path_helper = False

    self._extmethods = False
    self.__pluggable = YANGDynClass(base=YANGListType("pluggable_id",yc_pluggable_pluggable_config__pluggables_pluggable, yang_name="pluggable", parent=self, is_container='list', user_ordered=False, path_helper=self._path_helper, yang_keys='pluggable-id', extensions=None), is_container='list', yang_name="pluggable", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, extensions=None, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='list', is_config=True)

    load = kwargs.pop("load", None)
    if args:
      if len(args) > 1:
        raise TypeError("cannot create a YANG container with >1 argument")
      all_attr = True
      for e in self._pyangbind_elements:
        if not hasattr(args[0], e):
          all_attr = False
          break
      if not all_attr:
        raise ValueError("Supplied object did not have the correct attributes")
      for e in self._pyangbind_elements:
        nobj = getattr(args[0], e)
        if nobj._changed() is False:
          continue
        setmethod = getattr(self, "_set_%s" % e)
        if load is None:
          setmethod(getattr(args[0], e))
        else:
          setmethod(getattr(args[0], e), load=load)

  def _path(self):
    if hasattr(self, "_parent"):
      return self._parent._path()+[self._yang_name]
    else:
      return ['pluggables']

  def _get_pluggable(self):
    """
    Getter method for pluggable, mapped from YANG variable /pluggables/pluggable (list)

    YANG Description: List of pluggables and their configurations.
    """
    return self.__pluggable
      
  def _set_pluggable(self, v, load=False):
    """
    Setter method for pluggable, mapped from YANG variable /pluggables/pluggable (list)
    If this variable is read-only (config: false) in the
    source YANG file, then _set_pluggable is considered as a private
    method. Backends looking to populate this variable should
    do so via calling thisObj._set_pluggable() directly.

    YANG Description: List of pluggables and their configurations.
    """
    if hasattr(v, "_utype"):
      v = v._utype(v)
    try:
      t = YANGDynClass(v,base=YANGListType("pluggable_id",yc_pluggable_pluggable_config__pluggables_pluggable, yang_name="pluggable", parent=self, is_container='list', user_ordered=False, path_helper=self._path_helper, yang_keys='pluggable-id', extensions=None), is_container='list', yang_name="pluggable", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, extensions=None, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='list', is_config=True)
    except (TypeError, ValueError):
      raise ValueError({
          'error-string': """pluggable must be of a type compatible with list""",
          'defined-type': "list",
          'generated-type': """YANGDynClass(base=YANGListType("pluggable_id",yc_pluggable_pluggable_config__pluggables_pluggable, yang_name="pluggable", parent=self, is_container='list', user_ordered=False, path_helper=self._path_helper, yang_keys='pluggable-id', extensions=None), is_container='list', yang_name="pluggable", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, extensions=None, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='list', is_config=True)""",
        })

    self.__pluggable = t
    if hasattr(self, '_set'):
      self._set()

  def _unset_pluggable(self):
    self.__pluggable = YANGDynClass(base=YANGListType("pluggable_id",yc_pluggable_pluggable_config__pluggables_pluggable, yang_name="pluggable", parent=self, is_container='list', user_ordered=False, path_helper=self._path_helper, yang_keys='pluggable-id', extensions=None), is_container='list', yang_name="pluggable", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, extensions=None, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='list', is_config=True)

  pluggable = __builtin__.property(_get_pluggable, _set_pluggable)


  _pyangbind_elements = OrderedDict([('pluggable', pluggable), ])


class pluggable_config(PybindBase):
  """
  This class was auto-generated by the PythonClass plugin for PYANG
  from YANG module pluggable-config - based on the path /pluggable-config. Each member element of
  the container is represented as a class variable - with a specific
  YANG type.
  """
  __slots__ = ('_path_helper', '_extmethods', '__pluggables',)

  _yang_name = 'pluggable-config'
  _yang_namespace = 'urn:pluggable-config:1.0'

  _pybind_generated_by = 'container'

  def __init__(self, *args, **kwargs):

    self._path_helper = False

    self._extmethods = False
    self.__pluggables = YANGDynClass(base=yc_pluggables_pluggable_config__pluggables, is_container='container', yang_name="pluggables", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, extensions=None, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='container', is_config=True)

    load = kwargs.pop("load", None)
    if args:
      if len(args) > 1:
        raise TypeError("cannot create a YANG container with >1 argument")
      all_attr = True
      for e in self._pyangbind_elements:
        if not hasattr(args[0], e):
          all_attr = False
          break
      if not all_attr:
        raise ValueError("Supplied object did not have the correct attributes")
      for e in self._pyangbind_elements:
        nobj = getattr(args[0], e)
        if nobj._changed() is False:
          continue
        setmethod = getattr(self, "_set_%s" % e)
        if load is None:
          setmethod(getattr(args[0], e))
        else:
          setmethod(getattr(args[0], e), load=load)

  def _path(self):
    if hasattr(self, "_parent"):
      return self._parent._path()+[self._yang_name]
    else:
      return []

  def _get_pluggables(self):
    """
    Getter method for pluggables, mapped from YANG variable /pluggables (container)

    YANG Description: Container for pluggables.
    """
    return self.__pluggables
      
  def _set_pluggables(self, v, load=False):
    """
    Setter method for pluggables, mapped from YANG variable /pluggables (container)
    If this variable is read-only (config: false) in the
    source YANG file, then _set_pluggables is considered as a private
    method. Backends looking to populate this variable should
    do so via calling thisObj._set_pluggables() directly.

    YANG Description: Container for pluggables.
    """
    if hasattr(v, "_utype"):
      v = v._utype(v)
    try:
      t = YANGDynClass(v,base=yc_pluggables_pluggable_config__pluggables, is_container='container', yang_name="pluggables", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, extensions=None, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='container', is_config=True)
    except (TypeError, ValueError):
      raise ValueError({
          'error-string': """pluggables must be of a type compatible with container""",
          'defined-type': "container",
          'generated-type': """YANGDynClass(base=yc_pluggables_pluggable_config__pluggables, is_container='container', yang_name="pluggables", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, extensions=None, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='container', is_config=True)""",
        })

    self.__pluggables = t
    if hasattr(self, '_set'):
      self._set()

  def _unset_pluggables(self):
    self.__pluggables = YANGDynClass(base=yc_pluggables_pluggable_config__pluggables, is_container='container', yang_name="pluggables", parent=self, path_helper=self._path_helper, extmethods=self._extmethods, register_paths=True, extensions=None, namespace='urn:pluggable-config:1.0', defining_module='pluggable-config', yang_type='container', is_config=True)

  pluggables = __builtin__.property(_get_pluggables, _set_pluggables)


  _pyangbind_elements = OrderedDict([('pluggables', pluggables), ])


