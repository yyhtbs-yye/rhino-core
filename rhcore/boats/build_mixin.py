from copy import deepcopy

from rhcore.utils.build_components import build_module, build_modules, build_optimizer, build_lr_scheduler

class BuildMixin:

    def __init__(self, config):

        boat_config = config['boat']

        self.models = build_modules(boat_config['models'])
        self.pretrained = build_modules(boat_config.get('pretrained', {}))
        self.others = build_modules(boat_config.get('others', {}))
        self.processors = self.build_processors(boat_config.get('processors', {}))
        self.losses, self.loss_weight_schedulers = self.build_losses(boat_config['losses'])
        self.optimizers, self.gradient_clipping, self.lr_schedulers = self.build_optimizers(config['optimization'])

        self.metrics = build_modules(config['validation'].get('metrics', {}))

    def build_losses(self, configs):

        all_losses = {}
        all_loss_weight_schedulers = {}

        for name, config in configs.items():

            if 'weight' in config:
                w = float(config.pop('weight'))
                all_loss_weight_schedulers[name] = lambda _step: w
            elif 'weight_scheduler' in config:
                weight_scheduler_config = config.pop('weight_scheduler')
                all_loss_weight_schedulers[name] = build_module(weight_scheduler_config)
            else:
                all_loss_weight_schedulers[name] = lambda _step: 1.0

            new_module = build_module(config)

            if all_losses.get(name) is None or type(new_module) != type(all_losses[name]):
                all_losses[name] = new_module
        
        return all_losses, all_loss_weight_schedulers
    
    # Processors should support linking existing modules built in the boat.
    def build_processors(self, configs):
        processors = {}
        for name, config in configs.items():
            processors[name] = build_module(config)

            # Call context setter if exists, this pass the boat itself as context.
            if 'context_method' in config:
                _call_str = config.pop('context_method')
                if hasattr(processors[name], _call_str):
                    getattr(processors[name], _call_str) (self)
            else:
                pass
        return processors
    
    def build_optimizers(self, optimization_config):

        all_optimizers = {}
        all_gradient_clippings = {}
        all_lr_schedulers = {}
        
        for name in optimization_config:

            opt_config = deepcopy(optimization_config[name])

            all_gradient_clippings[name] = opt_config.pop('gradient_clipping', None)

            if 'bind_to' in opt_config:
                bind_to = opt_config.pop('bind_to', None)
                assert bind_to is not None
                if isinstance(bind_to, str):
                    bind_to = [bind_to] 

                all_params = []
                for b in bind_to:
                    assert b in self.models, f"Optimizer bind_to model '{b}' not found in boat models."
                    all_params.extend(list(self.models[b].parameters()))
                new_optimizer = build_optimizer(all_params, opt_config)
            else:
                new_optimizer = build_optimizer(self.models[name].parameters(), opt_config)

            # Handling the Load/Save from/to Pth with updated non-stateful parameters support. 
            if all_optimizers.get(name) is None or type(new_optimizer) != type(all_optimizers[name]):
                all_optimizers[name] = new_optimizer
            else:
                # same type and already have an optimizer (possibly with loaded state):
                # keep state, overwrite lr & other non-stateful hyperparams
                old_opt = all_optimizers[name]

                # if param group structure changed, safest is to replace
                if len(old_opt.param_groups) != len(new_optimizer.param_groups):
                    all_optimizers[name] = new_optimizer
                    continue

                for g_old, g_new in zip(old_opt.param_groups, new_optimizer.param_groups):
                    for k, v in g_new.items():
                        if k != "params":      # overwrite existing params with new ones, update lr, betas, weight_decay, etc.
                            g_old[k] = v
            
            # Create Learning Rate Schedulers
            if 'lr_scheduler' in optimization_config[name] and len(optimization_config[name].get('lr_scheduler', {})) > 0:
                new_lr_schelduler = build_lr_scheduler(all_optimizers[name], optimization_config[name].get('lr_scheduler', {}))
                if all_lr_schedulers.get(name) is None or type(new_lr_schelduler) != type(all_lr_schedulers[name]):
                    all_lr_schedulers[name] = new_lr_schelduler

        return all_optimizers, all_gradient_clippings, all_lr_schedulers
