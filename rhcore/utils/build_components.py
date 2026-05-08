import torch

import importlib
from functools import partial

class CallableWrapper:
    def __init__(self, func, **attrs):
        self.func = func
        self.__name__ = getattr(func, "__name__", self.__class__.__name__)
        self.__doc__ = getattr(func, "__doc__", None)

        for k, v in attrs.items():
            setattr(self, k, v)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

def get_class(path, name=None):
    """Dynamically load a class from a module."""

    if name is None:
        path, name = path.rsplit('.', 1)

    module = importlib.import_module(path)
    return getattr(module, name)

def build_module(config):

    if config is None:
        return None

    config_type = config.get('_type')
    if config_type is not None:
        config_type = str(config_type).strip()
        if config_type == 'func':
            if 'path' in config and 'name' in config:
                func = get_class(config['path'], config['name'])
            elif 'mpath' in config:
                func = get_class(config['mpath'])
            else:
                raise ValueError("Function configuration must contain ('path' and 'name') or ('mpath') key.")

            extra_configs = {k: config[k] for k in config if k[0] == '_'}
            return CallableWrapper(partial(func, **config.get('params', {})), **extra_configs)

        raise ValueError(f"Unsupported module _type: {config_type}")
    
    if 'path' in config and 'name' in config:
        """Build a module from configuration."""
        class_ = get_class(config['path'], config['name'])
    elif 'mpath' in config:
        class_ = get_class(config['mpath'])
    else:
        raise ValueError("Configuration must contain ('path' and 'name') or ('mpath') key.")
    
    load_pretrained = None

    if 'pretrained' in config and hasattr(class_, 'from_pretrained'):
        module = class_.from_pretrained(config['pretrained'])        
    elif 'config' in config and hasattr(class_, 'from_config'):
        module = class_.from_config(config['config'])
    elif 'config' in config and not hasattr(class_, 'from_config'):
        module = class_(config['config'])
    elif 'params' in config:
        if 'load_pretrained' in config['params']:
            load_pretrained = config['params'].pop('load_pretrained')
        module = class_(**config['params'])
        if load_pretrained is not None:
            module.load_state_dict(torch.load(load_pretrained, map_location="cpu"), strict=True)
    else:
        raise ValueError("Configuration must contain 'pretrained', 'config', or 'params' key.")

    if 'wrapper' in config:
        if isinstance(config['wrapper'], dict): # It is the case that the wrapper needs parameters
            wrapper_class_ = get_class(config['wrapper']['mpath'])
            module = wrapper_class_(module, **config['wrapper']['params'])
        else:                                   # It is the case that the wrapper DOES NOT need any parameter
            wrapper_class_ = get_class(config['wrapper'])
            module = wrapper_class_(module)

    return module

def build_modules(configs):
    """Build multiple modules from configuration."""
    modules = {}

    for k, v in configs.items():
        modules[k] = build_module(v)
    
    return modules

def build_optimizer(model_parameters, config):
    """Build a module from configuration."""
    class_ = get_class(config['path'], config['name'])
    
    if 'params' in config:
        return class_(model_parameters, **config['params'])
    else:
        raise ValueError("Configuration must contain 'params' key.")

def build_lr_scheduler(optimizer, config):
    """Build a module from configuration."""
    class_ = get_class(config['path'], config['name'])
    
    if 'params' in config:
        return class_(optimizer, **config['params'])
    else:
        raise ValueError("Configuration must contain 'params' key.")

def build_dataset(config):
    
    return build_module(config)

def build_logger(config):
    
    return build_module(config)
