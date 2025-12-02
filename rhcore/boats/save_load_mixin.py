from rhtrain.utils.load_save_utils import save_state, load_state

class SaveLoadMixin:

    def __init__(self):
        pass

    # ------------------------------------ State S/L ---------------------------------------------
    def save_state(self, run_folder, prefix="boat_state", global_step=None, epoch=None):
        return save_state(run_folder, prefix, boat=self, global_step=global_step, epoch=epoch)

    def load_state(self, state_path, strict=True):
        return load_state(state_path, boat=self, strict=strict)
