
from rhcore.boats.base_boat import BaseBoat

from rhcore.boats.base_boat import BaseBoat, _has_grad
from rhcore.boats.adapt_mixin import AdaptMixin

import warnings

class AdaptBoat(BaseBoat, AdaptMixin):

    def __init__(self, config={}):
        assert config is not None, "main config must be provided"
        super().__init__(config)

        self._adaptor_targets = {}       # adaptor_name -> target_model_name
        self._adapted_target_names = set()

        if 'adaptors' in self.trainer_config and self.trainer_config['adaptors']:
            # Normalize the adaptor config in the SAME way AdaptMixin does,
            # so names line up with self.adaptors keys.
            norm_cfg = self._normalize_adaptor_cfg(self.trainer_config['adaptors'])
            for adaptor_name, adaptor_cfg in norm_cfg.items():
                # do not mutate user config
                target_name = adaptor_cfg.get('target_model_name', None)
                if target_name is None:
                    raise KeyError(f"Adaptor '{adaptor_name}' missing 'target_model_name'")
                self._adaptor_targets[adaptor_name] = target_name
                self._adapted_target_names.add(target_name)

            # Build/wrap adaptors
            AdaptMixin.__init__(self, self.trainer_config['adaptors'])

            # Immediately enforce "train adaptor only" policy
            self._train_adaptors_only()

    # -----------------------------
    # Core logic: freeze base + train adaptors only
    # -----------------------------
    def _train_adaptors_only(self):
        """
        Freeze adapted target model(s) and force them to eval(),
        then enable training only for adaptor parts.
        """
        if not getattr(self, "adaptors", None):
            return

        # 1) Freeze + eval the adapted target model(s)
        for target_name in self._adapted_target_names:
            if not hasattr(self, "models") or target_name not in self.models:
                warnings.warn(f"Target model '{target_name}' not found in self.models; skipping freeze/eval.")
                continue

            target_model = self.models[target_name]
            target_model.eval()
            for p in target_model.parameters():
                p.requires_grad_(False)

        # 2) Ask adaptors to re-enable training for adaptor params ONLY
        for adaptor_name, adaptor in self.adaptors.items():
            target_name = self._adaptor_targets.get(adaptor_name, None)
            if target_name is None or not hasattr(self, "models") or target_name not in self.models:
                warnings.warn(f"Adaptor '{adaptor_name}' has no resolvable target; skipping adaptor-only enable.")
                continue

            target_model = self.models[target_name]
            self._enable_adaptor_training(adaptor, target_model)

    def _enable_adaptor_training(self, adaptor, target_model):
        """
        Compatibility shim: call the adaptor-provided method if present.
        Assumption: adaptor implements *some* method to (re)enable adaptor params
        and set adaptor submodules to train(), while base remains eval+frozen.
        """
        # Preferred/expected method name (from our earlier convention)
        candidate_methods = (
            "enable_adaptor_training",
            "train_adaptor_only",
            "set_train_adaptor_only",
            "setup_train_adaptor_only",
        )

        for m in candidate_methods:
            fn = getattr(adaptor, m, None)
            if callable(fn):
                fn(target_model)
                return

        # Fallback: best-effort. This may be insufficient if the adaptor injects
        # parameters into wrapped layers but does not expose them via adaptor.parameters().
        adaptor.train()
        any_param = False
        for p in adaptor.parameters(recurse=True):
            any_param = True
            p.requires_grad_(True)

        if not any_param:
            warnings.warn(
                "No known adaptor training-enablement method found, and adaptor has no parameters "
                "visible via adaptor.parameters(). Provide adaptor.enable_adaptor_training(target_model) "
                "(or one of the recognized aliases) to reliably unfreeze adaptor params inside wrappers."
            )

    # -----------------------------
    # Keep invariants even if training loop calls model.train()
    # -----------------------------
    def train(self, mode: bool = True):
        """
        If mode=True, keep base (adapted targets) in eval() and frozen, while adaptors are trainable.
        If mode=False, standard eval() for everything.
        """
        super().train(mode)
        if mode:
            self._train_adaptors_only()
        return self

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _normalize_adaptor_cfg(config):
        """
        Mirror AdaptMixin's normalization rules so adaptor names match self.adaptors keys.
        """
        if not config:
            return {}
        if isinstance(config, dict) and 'adaptor_module_config' in config:
            return {'default': config}
        if isinstance(config, (list, tuple)):
            return {str(idx): cfg for idx, cfg in enumerate(config)}
        if isinstance(config, dict):
            return config
        raise TypeError(f"Unsupported adaptors config type: {type(config)}")
