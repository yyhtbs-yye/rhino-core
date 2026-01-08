
import torch

from rhcore.boats.boat_template import BoatTemplate

from rhcore.boats.utils.aggregation import list_of_dict_aggr_to_dict
from rhcore.boats.utils.split_batch import split_batch
from rhtrain.utils.ddp_utils import ddp_no_sync_all, move_to_device
from rhcore.boats.ema_mixin import EMAMixIn
from rhcore.boats.log_mixin import LogMixIn
from rhcore.boats.build_mixin import BuildMixin
from rhcore.boats.save_load_mixin import SaveLoadMixin

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

class BaseBoat(BoatTemplate, EMAMixIn, LogMixIn, BuildMixin, SaveLoadMixin):

    def __init__(self, config={}):

        assert config is not None, "main config must be provided"
        super().__init__(config)

        self.logging_config = config.get('logging', {})
        self.ema_config = config.get('ema', {})
        self.visualization_config = config.get('visualization', {})
        self.trainer_config = config['trainer']
        self.total_micro_steps = config['optimization'].pop('total_micro_steps', 1)

        BuildMixin.__init__(self, config)

        if config['rank'] == 0:
            LogMixIn.__init__(self, self.logging_config, self.visualization_config)

        if self.ema_config:
            EMAMixIn.__init__(self, self.ema_config)

        self.ordered_groups = config['boat'].get('ordered_groups', None)
        if self.ordered_groups is None or len(self.ordered_groups) == 0:
            self.ordered_groups = [
                {
                    'boat_loss_method_str': 'training_calc_losses',
                    'target_loss_name': 'total_loss',
                    'models': ['net'],
                    'optimizers': ['net'],
                    'train_interval': 1,
                },
            ]
        
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

    def training_gradient_descent(self, scaler, active_optimizers):

        if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
            # Unscale once per optimizer before clipping
            for name, opt in active_optimizers.items():
                scaler.unscale_(opt)

        # Now gradients are unscaled (or were never scaled); safe to clip
        for name, opt in active_optimizers.items():
            max_norm = self.gradient_clipping.get(name, None)
            if max_norm:
                for group in opt.param_groups:
                    torch.nn.utils.clip_grad_norm_(group["params"], max_norm)

        # Step
        if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
            for opt in active_optimizers.values():
                scaler.step(opt)
            scaler.update()
        else:
            for opt in active_optimizers.values():
                opt.step()

    def training_lr_scheduling_step(self, active_optimizer_names):
        
        for k in active_optimizer_names: 
            if k in self.lr_schedulers:
                self.lr_schedulers[k].step()

    def train_a_group(self, group_config, batch, batch_idx, epoch, *, scaler=None):

        train_interval = group_config.get('train_interval', 1)

        if not ((train_interval > 0) and (batch_idx % train_interval == 0)):
            return {}

        boat_loss_method_str = group_config['boat_loss_method_str']
        boat_loss_method = getattr(self, boat_loss_method_str)

        target_loss_name = group_config['target_loss_name']

        active_models = {name: self.models[name] for name in group_config['models']}
        active_optimizers = {name: self.optimizers[name] for name in group_config['optimizers']}
        
        # Preparation: Freeze all other models, zero_grad active_optimizer 
        self._freeze_all_except(active_models)
        self._zero_grad(active_optimizers, set_to_none=True)

        micro_batches = split_batch(batch, self.total_micro_steps)

        micro_losses_list = []
        for current_micro_step, micro_batch in enumerate(micro_batches):
            micro_batch = move_to_device(micro_batch, self.device)
            micro_losses = boat_loss_method(micro_batch)
            micro_losses_list.append(micro_losses)

            if isinstance(target_loss_name, str):
                micrompathloss = micro_losses[target_loss_name] / self.total_micro_steps
            elif isinstance(target_loss_name, list) or isinstance(target_loss_name, tuple):
                micrompathloss = [micro_losses[k] / self.total_micro_steps for k in target_loss_name]

            self.training_backpropagation(micrompathloss, current_micro_step, scaler)

        self.training_gradient_descent(scaler, active_optimizers)
        self.training_lr_scheduling_step(active_optimizers)

        return list_of_dict_aggr_to_dict(micro_losses_list)

    def training_all(self, batch, batch_idx, epoch, *, scaler=None):

        all_loss_dict = {}
        for group_config in self.ordered_groups:
            loss_dict = self.train_a_group(group_config, batch=batch, batch_idx=batch_idx, epoch=epoch, scaler=scaler)
            all_loss_dict.update(loss_dict)

        # EMA update (if enabled)
        if self.ema_config:
            self.update_ema()

        return all_loss_dict

    def calc_metrics(self, valid_output):

        results = {}
        for metric_name, metric in self.metrics.items():
            metric_val = metric(valid_output)
            if isinstance(metric_val, dict):
                results.update(metric_val)
            else:
                results[metric_name] = metric_val

        return results

    def maybe_get_ema_model(self, model_name):
        if hasattr(self, 'ema_models'):
            if model_name in self.ema_models:
                return self.ema_models[model_name]
            else:
                return self.models[model_name]
        else:
            return self.models[model_name]
        
