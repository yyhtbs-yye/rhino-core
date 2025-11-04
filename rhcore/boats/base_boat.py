from copy import deepcopy

import torch
from functools import partial

from rhcore.boats.boat_template import BoatTemplate
from rhcore.utils.build_components import (build_module, build_modules, build_optimizer, 
                                        build_lr_scheduler, build_logger)
from rhcore.loggers.helpers.base_image_visualizer import BaseImageVisualizer
from rhtrain.utils.ddp_utils import ddp_no_sync_all, move_to_device
from rhtrain.utils.load_save_utils import save_state, load_state

import warnings

def _has_grad(t):
    """
    Returns True if `t` participates in autograd:
    - requires_grad=True (leaf or non-leaf), or
    - has a grad_fn (non-leaf produced by ops on tensors requiring grad)
    """
    if not isinstance(t, torch.Tensor):
        return False
    return t.requires_grad or (t.grad_fn is not None)

class BaseBoat(BoatTemplate):

    def __init__(self, config={}):

        assert config is not None, "main config must be provided"

        self.boat_config = config.get('boat', {})
        self.optimization_config = config.get('optimization', {})
        self.validation_config = config.get('validation', {})
        self.visualization_config = config.get('visualization', {})
        self.logging_config = config.get('logging', {})
        self.trainer_config = config.get('trainer', {})

        self.total_micro_steps = self.optimization_config.pop('total_micro_steps', 1)
        self.target_loss_key = self.optimization_config.pop('target_loss_key', 'total_loss')

        self.models = {}
        self.pretrained = {}
        self.losses = {}
        self.optimizers = {}
        self.lr_schedulers = {}
        self.metrics = {}
        self.loggers = {}
        self.viz = None

        self.build_models()
        self.build_pretrained()
        self.build_losses()
        self.build_optimizers()
        self.build_metrics()
        self.build_others()

        if config['rank'] == 0:
            self.build_loggers()
            first_logger = next(iter(self.loggers.values()))

            if (self.visualization_config.get('save_images', False) 
                or self.trainer_config.get('save_images', False)):
                
                self.viz = BaseImageVisualizer(first_logger, 
                                            wnb=self.visualization_config.get('wnb', (0.5, 0.5)), 
                                            max_images=self.visualization_config.get('max_images', 4),
                                            dataformats=self.visualization_config.get('dataformats', 'CHW'))

        self.use_ema = self.optimization_config.get('use_ema', False)
        if self.use_ema:
            self._setup_ema()
            self.ema_start = self.optimization_config.get('ema_start', 0)


    def to(self, device):
        """
        Move all models and metrics to the specified device.
        
        Args:
            device: The device to move the models to (e.g., 'cuda:3', 'cpu')
            
        Returns:
            self: The boat with models on the specified device
        """
        self.device = device
        for name, model in self.models.items():
            if hasattr(model, 'to'):
                self.models[name] = model.to(device)
        
        if hasattr(self, 'losses'):
            for name, loss in self.losses.items():
                if hasattr(loss, 'to'):
                    self.losses[name] = loss.to(device)
        
        if hasattr(self, 'metrics'):
            for name, metric in self.metrics.items():
                if hasattr(metric, 'to'):
                    self.metrics[name] = metric.to(device)
        
        if hasattr(self, 'pretrained'):
            for name, model in self.pretrained.items():
                if hasattr(model, 'to'):
                    self.pretrained[name] = model.to(self.device)
        
        # Move optimizer states to the same device
        self.move_optimizer_to_device(device)
        return self

    def parameters(self):
        for model_name, model in self.models.items():
            if hasattr(model, 'parameters'):
                for param in model.parameters():
                    yield param    

    def train(self):
        """
        Set all models to training mode.
        
        Returns:
            self
        """
        for model_name, model in self.models.items():
            if hasattr(model, 'train'):
                model.train()
        return self
    
    def eval(self):
        """
        Set all models to evaluation mode.
        
        Returns:
            self
        """
        for model_name, model in self.models.items():
            if hasattr(model, 'eval'):
                model.eval()
        return self

    # ------------------------------------ Training Step ---------------------------------------------

    def training_backpropagation(self, losses, current_micro_step, scaler):
        if not isinstance(losses, (list, tuple)):
            losses = [losses]

        use_no_sync = (self.total_micro_steps > 1) and (current_micro_step < self.total_micro_steps - 1)

        with ddp_no_sync_all(self, enabled=use_no_sync):
            for idx, loss in enumerate(losses):
                # Skip non-tensors entirely
                if not isinstance(loss, torch.Tensor):
                    warnings.warn(f"[micro {current_micro_step}] loss[{idx}] is not a Tensor; skipping backward.")
                    continue

                # Optional sanity checks (won't block valid zero losses with real grad paths)
                if not _has_grad(loss):
                    # This covers constants like torch.tensor(0.) with requires_grad=False,
                    # or tensors detached from the graph.
                    warnings.warn(f"[micro {current_micro_step}] loss[{idx}] has no grad path; skipping backward.")
                    continue

                if not torch.isfinite(loss):
                    warnings.warn(f"[micro {current_micro_step}] loss[{idx}] is not finite ({loss}); skipping backward.")
                    continue

                # If it's not scalar, reduce to a scalar to be safe (mean is typical)
                if loss.ndim != 0:
                    loss = loss.mean()

                if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()


    def training_gradient_descent(self, scaler, active_keys):

        if scaler is not None and scaler.is_enabled():
            for k in active_keys: scaler.step(self.optimizers[k])
            scaler.update()
        else:
            for k in active_keys: self.optimizers[k].step()

    def training_lr_scheduling_step(self, active_keys):

        for k in active_keys: 
            if k in self.lr_schedulers:
                self.lr_schedulers[k].step()
            
    def training_step(self, batch, batch_idx, epoch, *, scaler=None):
        
        active_keys = list(self.optimizers.keys())
        
        micro_batches = self._split_batch(batch, self.total_micro_steps)

        self._zero_grad(active_keys, set_to_none=True)

        micro_losses_list = []
        for current_micro_step, micro_batch in enumerate(micro_batches):
            micro_batch = move_to_device(micro_batch, self.device)
            micro_losses = self.training_calc_losses(micro_batch)
            micro_losses_list.append(micro_losses)

            if isinstance(self.target_loss_key, str):
                micrompathloss = micro_losses[self.target_loss_key] / self.total_micro_steps
            elif isinstance(self.target_loss_key, list) or isinstance(self.target_loss_key, tuple):
                micrompathloss = [micro_losses[k] / self.total_micro_steps for k in self.target_loss_key]

            self.training_backpropagation(micrompathloss, current_micro_step, scaler)

        self.training_gradient_descent(scaler, active_keys)
        
        self._update_ema()

        self.training_lr_scheduling_step(active_keys)

        return self._aggregate_loss_dicts(micro_losses_list)

    # ------------------------------------ Visualization ---------------------------------------------

    def visualize_step(self, named_imgs, batch_idx):

        if self.viz is None:
            return

        """Visualize validation results."""
        if self.visualization_config.get('first_batch_only', True) and batch_idx == 0:
            # Limit the number of samples to visualize
            for key in named_imgs.keys():
                if named_imgs[key].shape[0] > self.visualization_config.get('num_vis_samples', 4):
                    named_imgs[key] = named_imgs[key][:self.visualization_config.get('num_vis_samples')]
            
            # Log visualizations to the experiment tracker
            self.viz(
                images_dict=named_imgs,
                keys=list(named_imgs.keys()),
                global_step=self.get_global_step(),
                prefix='val',
                texts='texts',
            )

    # ------------------------------------ Result Logging ---------------------------------------------

    def _log_values(self, results, prefix=''):
        for _, logger in self.loggers.items():
            logger.log_metrics(results, step=self.get_global_step(), prefix=prefix)

    def _log_value(self, result, metric_name, prefix=''):
        for _, logger in self.loggers.items():
            logger.log_metrics({metric_name: result}, step=self.get_global_step(), prefix=prefix)

    def log_train_losses(self, losses):
        for loss_name, loss_value in losses.items():
            self._log_value(loss_value.detach(), metric_name=loss_name, prefix='train')

    def log_valid_metrics(self, metrics):
        for metric_name, metric_value in metrics.items():
            self._log_value(metric_value.detach(), metric_name=metric_name, prefix='valid')

    # ------------------------------------ Global Step Mgmt ---------------------------------------------

    def get_global_step(self):
        return self.global_step()
    
    def attach_global_step(self, global_step):
        self.global_step = global_step

    # ------------------------------------ Build Component ---------------------------------------------
    def build_models(self):
        for model_name in self.boat_config.get('models', {}):
            new_module = build_module(self.boat_config['models'][model_name])
            if self.models.get(model_name) is None or type(new_module) != type(self.models[model_name]):
                self.models[model_name] = new_module

    def build_pretrained(self):
        for model_name in self.boat_config.get('pretrained', {}):
            new_module = build_module(self.boat_config['pretrained'][model_name])
            if self.pretrained.get(model_name) is None or type(new_module) != type(self.pretrained[model_name]):
                self.pretrained[model_name] = new_module

    def build_losses(self):
        for loss_name in self.boat_config.get('losses', {}):
            new_module = build_module(self.boat_config['losses'][loss_name])
            if self.losses.get(loss_name) is None or type(new_module) != type(self.losses[loss_name]):
                self.losses[loss_name] = new_module
                
    def build_metrics(self):
        self.metrics = build_modules(self.validation_config.get('metrics', {}))

    def build_optimizers(self):
        for opt_name in self.optimization_config:
            if 'ema' in opt_name:
                continue
            elif 'bind_to' in self.optimization_config[opt_name]:
                bind_to = self.optimization_config[opt_name].pop('bind_to', None)
                assert bind_to is not None
                if isinstance(bind_to, str):
                    bind_to = [bind_to] 

                all_params = []
                for b in bind_to:
                    assert b in self.models, f"Optimizer bind_to model '{b}' not found in boat models."
                    all_params.extend(list(self.models[b].parameters()))
                new_optimizer = build_optimizer(
                    all_params,
                    self.optimization_config[opt_name]
                )
            else:
                new_optimizer = build_optimizer(
                    self.models[opt_name].parameters(), 
                    self.optimization_config[opt_name]
                )
            if self.optimizers.get(opt_name) is None or type(new_optimizer) != type(self.optimizers[opt_name]):
                self.optimizers[opt_name] = new_optimizer

        self.build_lr_scheduler_by_name(opt_name)

    def build_lr_scheduler_by_name(self, model_name):

        if 'lr_scheduler' in self.optimization_config[model_name] and len(self.optimization_config[model_name].get('lr_scheduler', {})) > 0:
            new_lr_schelduler = build_lr_scheduler(self.optimizers[model_name], self.optimization_config[model_name].get('lr_scheduler', {}))
            if self.lr_schedulers.get(model_name) is None or type(new_lr_schelduler) != type(self.lr_schedulers[model_name]):
                self.lr_schedulers[model_name] = new_lr_schelduler
    
    def build_loggers(self):
        self.loggers = {}
        for logger_name in self.logging_config['loggers']:
            self.loggers[logger_name] = build_logger(self.logging_config['loggers'][logger_name])

    # ------------------------------------ EMA ---------------------------------------------
    def _setup_ema(self):
        """Set up Exponential Moving Average (EMA) model."""

        if isinstance(self.use_ema, bool):
            self.use_ema = {'ema_decay': 0.999, 'ema_start': 1000}

        if self.use_ema:
            self.models['net_ema'] = deepcopy(self.models['net'])
            for param in self.models['net_ema'].parameters():
                param.requires_grad = False
    
        self.ema_decay = self.use_ema.get('ema_decay', 0.999)
        self.ema_start = self.use_ema.get('ema_start', 1000)

    def _update_ema(self):

        if self.use_ema and self.get_global_step() >= self.ema_start:
            """Update EMA model parameters."""
            if not self.use_ema: return

            for ema_param, param in zip(self.models['net_ema'].parameters(), self.models['net'].parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
            for ema_buffer, buffer in zip(self.models['net_ema'].buffers(), self.models['net'].buffers()):
                ema_buffer.data.copy_(buffer.data)

    # ------------------------------------ State S/L ---------------------------------------------
    def save_state(self, run_folder, prefix="boat_state", global_step=None, epoch=None):
        return save_state(run_folder, prefix, boat=self, global_step=global_step, epoch=epoch)

    def load_state(self, state_path, strict=True):
        return load_state(state_path, boat=self, strict=strict)

    def move_optimizer_to_device(self, device):        # DO NOT edit optim.param_groups here
        for _, optim in self.optimizers.items():
            # move state tensors
            for state in optim.state.values():
                for k, v in list(state.items()):
                    if torch.is_tensor(v):
                        state[k] = v.to(device)

    # ------------------------------------ Metrics ---------------------------------------------
    def _reset_metrics(self):
        for _, metric in self.metrics.items(): # _: metric_name
            if hasattr(metric, 'reset'):
                metric.reset()

    def _calc_metrics(self, valid_output):

        results = {}
        for metric_name, metric in self.metrics.items():
            metric_val = metric(valid_output)
            if isinstance(metric_val, dict):
                results.update(metric_val)
            else:
                results[metric_name] = metric_val

        return results

    # ------------------------------------ Utilities ---------------------------------------------
    def _zero_grad(self, active_keys, set_to_none=True):
        for key in active_keys:
            self.optimizers[key].zero_grad(set_to_none=set_to_none)

    def _split_batch(self, x, parts):
        # returns a list of length `parts` mirroring x's structure
        if torch.is_tensor(x):
            # torch.chunk handles non-divisible sizes
            return list(torch.chunk(x, parts, dim=0))
        if isinstance(x, dict):
            per = [dict() for _ in range(parts)]
            for k, v in x.items():
                chunks = self._split_batch(v, parts)
                for i in range(parts):
                    per[i][k] = chunks[i]
            return per
        if isinstance(x, (list, tuple)):
            elems = [self._split_batch(v, parts) for v in x]
            out = []
            for i in range(parts):
                out.append(type(x)(chunks[i] for chunks in elems))
            return out
        # non-tensor leaf: replicate reference (ok for e.g. scalars/strings)
        return [x for _ in range(parts)]
    
    def _aggregate_loss_dicts(self, loss_dicts):
        """
        Simple (unweighted) mean of each key across a list of micro-batch loss dicts.
        Missing keys are ignored for that key's average.
        Tensor values are detached and converted to floats.
        """
        if not loss_dicts:
            return {}

        keys = set().union(*(d.keys() for d in loss_dicts))
        out = {}

        for k in keys:
            vals = []
            for d in loss_dicts:
                if k not in d or d[k] is None:
                    continue
                v = d[k]
                if torch.is_tensor(v):
                    v = v.detach()
                    v = v.mean() if v.ndim > 0 else v
                else:
                    v = v
                vals.append(v)
            if vals:
                out[k] = sum(vals) / len(vals)

        return out

    def build_others(self):
        pass

    def _install_forward_hooks(self, model_layer_names={}, hook_fn=None):
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

    def _collect_from_forward_hooks(self, batch, batch_idx):
        if 'hook_fx' not in batch:
            batch['hook_fx'] = {}
        for model_name, memory in self.hook_memories.items():
            for layer_name, output in memory.items():
                if output is not None:
                    batch['hook_fx'][f"{model_name}_{layer_name}"] = output
        return batch
    