class Chainer:
    """Generic pipeline: just a list of callables.
    Each callable must support x = f(x, *args) or f(x, **kwargs)."""

    def __init__(self, module_configs):
        self.module_configs = module_configs
        self.modules = {}

    def set_context(self, context):
        self.modules = {}

        for i, module_config in enumerate(self.module_configs):
            # e.g. 'pretrained.latent_encoder' -> key1='pretrained', key2='latent_encoder'
            key1, key2 = module_config['ref'].split('.')

            self.modules[i] = {
                'module': getattr(context, key1)[key2],
                'args': module_config.get('args', []),
                'method': module_config.get('method', None),
            }

    def __call__(self, x):
        # IMPORTANT: iterate over the stored dict values, not the keys
        for i in range(len(self.modules)):
            module = self.modules[i]['module']
            method_name = self.modules[i]['method']
            args = self.modules[i]['args']

            # If a method is specified, call that method on the module
            if method_name is not None and hasattr(module, method_name):
                module = getattr(module, method_name)

            # Call with args in whatever form they are
            if isinstance(args, list):
                x = module(x, *args)
            elif isinstance(args, dict):
                x = module(x, **args)
            else:
                x = module(x, args)

        return x


if __name__ == "__main__":

    # ---- Dummy context to test with ----

    class DummyContext:
        def __init__(self):
            # This matches the "pretrained.latent_encoder" lookup:
            # getattr(context, "pretrained")["latent_encoder"]
            self.pretrained = {
                "latent_encoder": self.latent_encoder
            }
            # This matches "others.fmt"
            self.others = {
                "fmt": self.fmt
            }

        # Example module 1
        def latent_encoder(self, x, suffix):
            # Just append something to show it was called
            return f"{x}<{suffix}>"

        # Example module 2
        def fmt(self, x):
            # Wrap in brackets to show second step
            return f"[{x}]"

    chainer = Chainer(
        {"name": "pretrained.latent_encoder", "args": ["encode"]},
        {"name": "others.fmt"}  # no args -> uses default []
    )

    ctx = DummyContext()
    chainer.set_context(ctx)

    x0 = "input"
    y = chainer(x0)

    print("Input :", x0)
    print("Output:", y)
