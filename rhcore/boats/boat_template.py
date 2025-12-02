from abc import ABC, abstractmethod
import torch

class BoatTemplate(ABC):
    """
    Abstract base class for model containers to be used with the Trainer.
    
    A "Boat" represents a container for models, optimizers, and training logic,
    but is not itself a nn.Module.
    """

    @abstractmethod
    def __init__(self, config):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    def to(self, device):
        """
        Move all models and metrics to the specified device.
        
        Args:
            device: The device to move the models to (e.g., 'cuda:3', 'cpu')
            
        Returns:
            self: The boat with models on the specified device
        """

        def move_optimizer_to_device(optimizers, device):        # DO NOT edit optim.param_groups here
            for _, optim in optimizers.items():
                # move state tensors
                for state in optim.state.values():
                    for k, v in list(state.items()):
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
        self.device = device
        for name, model in self.models.items():
            if hasattr(model, 'to'):
                self.models[name] = model.to(device)

        if hasattr(self, 'ema_models'):
            for name, model in self.ema_models.items():
                if hasattr(model, 'to'):
                    self.ema_models[name] = model.to(device)

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
        move_optimizer_to_device(self.optimizers, device)
        return self

    def parameters(self):
        for model_name, model in self.models.items():
            if hasattr(model, 'parameters'):
                for param in model.parameters():
                    yield param    

    def train(self): # Set all models to training mode.

        for model_name, model in self.models.items():
            if hasattr(model, 'train'):
                model.train()
        return self
    
    def eval(self): # Set all models to evaluation mode.

        for model_name, model in self.models.items():
            if hasattr(model, 'eval'):
                model.eval()
        return self
    
    @abstractmethod
    def training_lr_scheduling_step(self):
        pass

    @abstractmethod
    def training_backpropagation(self, loss, current_micro_step, scaler):
        pass

    @abstractmethod
    def training_gradient_descent(self, scaler, active_keys):
        pass

    @abstractmethod
    def train_a_group(self, group_config, batch, batch_idx, epoch, *, scaler):
        pass

    @abstractmethod
    def training_all(self, batch, batch_idx, epoch, *, scaler):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx, epoch):
        pass

    def get_global_step(self):
        return self.global_step()
    
    def attach_global_step(self, global_step):
        self.global_step = global_step

    # ------------------------------------ Gradient Management -------------------------
    def _unfreeze_all(self):
        for name in self.models:
            self.models[name].requires_grad_(True)

    def _freeze_all(self):
        for name in self.models:
            self.models[name].requires_grad_(False)

    def _freeze_all_except(self, active_models):
        for name in self.models:
            model = self.models[name]
            if not hasattr(model, 'requires_grad_'):
                continue
            if name in active_models:
                model.requires_grad_(True)
            else:
                model.requires_grad_(False)

    def _zero_grad(self, active_keys, set_to_none=True):
        for key in active_keys:
            self.optimizers[key].zero_grad(set_to_none=set_to_none)

