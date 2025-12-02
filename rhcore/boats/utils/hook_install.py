from functools import partial

def install_forward_hooks(self, model_layer_names={}, hook_fn=None):
    for model_name in model_layer_names:

        if model_name not in self.hook_memories:
            self.hook_memories[model_name] = {}

        if model_name not in self.models:
            continue
        
        layer_names = model_layer_names[model_name]
        if isinstance(layer_names, str):
            layer_names = [layer_names]

        for layer_name in layer_names:
            hook_handle = self.models[model_name].get_submodule(layer_name).register_forward_hook(
                partial(hook_fn, layer_name, self.hook_memories[model_name])
            )

def collect_from_forward_hooks(self, batch, batch_idx):
    if 'hook_fx' not in batch:
        batch['hook_fx'] = {}
    for model_name, memory in self.hook_memories.items():
        for layer_name, output in memory.items():
            if output is not None:
                batch['hook_fx'][f"{model_name}_{layer_name}"] = output
    return batch
