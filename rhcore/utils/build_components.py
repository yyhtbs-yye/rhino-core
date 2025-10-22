import importlib

def get_class(path, name=None):
    """Dynamically load a class from a module."""

    if name is None:
        path, name = path.rsplit('.', 1)

    module = importlib.import_module(path)
    return getattr(module, name)

def build_module(config):

    if config is None:
        return None
    
    if 'path' in config and 'name' in config:
        """Build a module from configuration."""
        class_ = get_class(config['path'], config['name'])
    elif 'mpath' in config:
        class_ = get_class(config['mpath'])
    else:
        raise ValueError("Configuration must contain 'path' and 'name' or 'mpath' key.")
    
    if 'pretrained' in config and hasattr(class_, 'from_pretrained'):
        module = class_.from_pretrained(config['pretrained'])
    elif 'config' in config and hasattr(class_, 'from_config'):
        module = class_.from_config(config['config'])
    elif 'config' in config and not hasattr(class_, 'from_config'):
        module = class_(config['config'])
    elif 'params' in config:
        module = class_(**config['params'])
    else:
        raise ValueError("Configuration must contain 'pretrained', 'config', or 'params' key.")

    if 'wrapper' in config:
        if isinstance(config['wrapper'], dict):
            wrapper = get_class(config['wrapper']['mpath'])
            module = wrapper(module, **config['wrapper']['params'])
        else:
            wrapper = get_class(config['wrapper'])
            module = wrapper(module)

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
