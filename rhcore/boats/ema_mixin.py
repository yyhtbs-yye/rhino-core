from copy import deepcopy
import torch
class EMAMixIn:

    def __init__(self, ema_config_dict):
        self.ema_models = {}
        self.ema_config_dict = ema_config_dict

        self.use_ema = False

        for name, config in ema_config_dict.items():
            assert name in self.models, f"EMA target '{name}' not found in self.models"

            ema_model = deepcopy(self.models[name])
            ema_model.requires_grad_(False)
            ema_model.ema_decay = config.get('ema_decay', 0.999)
            ema_model.ema_start = config.get('ema_start', 0)

            self.ema_models[name] = ema_model

            self.use_ema = True

    @torch.no_grad()
    def update_ema(self):
        for name in self.ema_config_dict:
            ema_model = self.ema_models[name]
            model = self.models[name]

            if self.get_global_step() < self.ema_config_dict[name].get('ema_start', 0):
                self.ema_models[name] = self.models[name]
                continue

            decay = ema_model.ema_decay

            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.mul_(decay).add_(param, alpha=1-decay)

            for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
                ema_buffer.copy_(buffer)
