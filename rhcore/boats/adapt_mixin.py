from copy import deepcopy

from rhcore.utils.build_components import build_module

class AdaptMixin:

    def __init__(self, config):
        # example config:{
        #   'adaptor_name_1': {
        #     'target_model_name': 'net',
        #     'adaptor_module_config': 
        #       {'path': '',
        #        'name': '',
        #        'params': {}},  # config for building adaptor module
        #     'layers': ['layer1', 'layer2']
        #   },
        #   ...
        # }

        self.adaptors = {}
        if not config:
            return

        for name, adaptor_config in config.items():
            adaptor_config = deepcopy(adaptor_config)

            target_model_name = adaptor_config.pop('target_model_name')
            layers = adaptor_config.pop('layers', None)
            adaptor_module_config = adaptor_config.pop('adaptor_module_config')

            assert target_model_name in self.models, f"Adaptor target '{target_model_name}' not found in self.models"
            target_model = self.models[target_model_name]

            adaptor = build_module(adaptor_module_config)

            assert layers is not None, f"Adaptor '{name}' must specify 'layers' to adapt"
            assert len(layers) > 0, f"Adaptor '{name}' must specify at least one layer to adapt"
            
            if isinstance(layers, str):
                layers = [layers]
            for layer_name in layers:
                self._setup_layer_adaptor(adaptor, target_model, layer_name)

            self.adaptors[name] = adaptor

    def _get_submodule(self, module, layer_name):
        current = module
        for name in layer_name.split('.'):
            if hasattr(current, "_modules") and name in current._modules:
                current = current._modules[name]
            else:
                current = getattr(current, name)
        return current

    def _set_submodule(self, module, layer_name, new_module):
        parts = layer_name.split('.')
        parent = module
        for name in parts[:-1]:
            if hasattr(parent, "_modules") and name in parent._modules:
                parent = parent._modules[name]
            else:
                parent = getattr(parent, name)
        last = parts[-1]
        if hasattr(parent, "_modules") and last in parent._modules:
            parent._modules[last] = new_module
        else:
            setattr(parent, last, new_module)

    def _setup_adaptor(self, adaptor, layer):
        if not hasattr(adaptor, 'setup'):
            raise AttributeError("adaptor_module_config must implement setup()")
        return adaptor.setup(layer)

    def _setup_layer_adaptor(self, adaptor, target_model, layer_name):
        layer = self._get_submodule(target_model, layer_name)
        wrapped = self._setup_adaptor(adaptor, layer)
        if wrapped is not None and wrapped is not layer:
            self._set_submodule(target_model, layer_name, wrapped)
    
