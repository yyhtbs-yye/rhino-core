import os
from PIL import Image
import numpy as np
import torch

class LoadVideoFromFolder:
    """Load a video from a folder of frames and convert to tensor.

    Expects in `results`:
        results[f"{key}_path"] = path_to_folder   (or list of frame paths)

    Produces in `results`:
        results[key]             = video array/tensor (T, C, H, W) if to_tensor else (T, H, W, C)
        results[f"{key}_ori_shape"] = video.shape
    """

    def __init__(
        self,
        keys,
        use_long: bool = False,
        to_tensor: bool = True,
        frame_extensions=None,
        sort_frames: bool = True,
    ):
        # Support both single key or list of keys
        if isinstance(keys, str):
            self.keys = [keys]
        else:
            self.keys = keys

        self.use_long = use_long
        self.to_tensor = to_tensor
        self.sort_frames = sort_frames

        if frame_extensions is None:
            frame_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        # normalize extensions to lower case
        self.frame_extensions = {ext.lower() for ext in frame_extensions}

    def _get_frame_files_from_folder(self, folder_path: str):
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Video folder does not exist: {folder_path}")

        files = os.listdir(folder_path)
        frame_files = [
            os.path.join(folder_path, f)
            for f in files
            if os.path.splitext(f)[1].lower() in self.frame_extensions
        ]
        if self.sort_frames:
            frame_files.sort()
        return frame_files

    def __call__(self, results):
        """Load video frames from folder (or list of frame paths) and convert to tensor."""
        for key in self.keys:
            src = results[f"{key}_path"]

            # Allow either a folder path or an explicit list of frame paths
            if isinstance(src, (list, tuple)):
                frame_files = list(src)
                if self.sort_frames:
                    frame_files.sort()
            else:
                frame_files = self._get_frame_files_from_folder(src)

            if not frame_files:
                raise ValueError(f"No frame images found for key '{key}' at {src}")

            frames = []
            for fp in frame_files:
                try:
                    img = Image.open(fp).convert("RGB")
                    frames.append(np.array(img))
                except Exception as e:
                    raise Exception(f"Failed to load frame: {fp}, {e}")

            # Check all frames have the same shape
            first_shape = frames[0].shape
            for i, arr in enumerate(frames):
                if arr.shape != first_shape:
                    raise ValueError(
                        f"Inconsistent frame shape in video for key '{key}': "
                        f"{frame_files[i]} has shape {arr.shape}, expected {first_shape}"
                    )

            # Stack into (T, H, W, C)
            video = np.stack(frames, axis=0)

            # Convert to tensor if requested
            if self.to_tensor:
                # (T, H, W, C) -> (T, C, H, W)
                video = video.transpose(0, 3, 1, 2)
                if self.use_long:
                    video = torch.from_numpy(video).long()
                else:
                    video = torch.from_numpy(video).float() / 255.0

            # Store in results
            results[key] = video
            results[f"{key}_ori_shape"] = video.shape

        return results
