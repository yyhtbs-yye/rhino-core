import os
from pathlib import Path
from torch.utils.data import Dataset
from rhcore.data.datasets import transforms

class BasicVideoDataset(Dataset):
    """Paired video dataset.

    - Each video is represented by a subfolder containing frame images (e.g. PNGs).
    - For each key in `folder_paths`, we look under:
        folder_paths[key] / data_prefix.get(key, "")
      and treat each subdirectory there as a video.
    - Videos are matched across keys by subfolder name.
    - Each sample is a dict:
        {
            "<key>_path": [list_of_frame_paths_for_that_video_for_this_key],
            ...
        }
    """

    def __init__(self, **dataset_config):
        super().__init__()

        # Config
        self.folder_paths = dataset_config.get("folder_paths", {})
        self.data_prefix = dataset_config.get("data_prefix", {})
        self.max_dataset_size = dataset_config.get("max_dataset_size", None)
        self.inflate_factor = dataset_config.get("inflate_factor", 1)

        if not self.folder_paths:
            raise ValueError("No folder_paths specified for BasicVideoDataset")

        # Scan videos
        self.video_items = self._scan_videos()

        # Build pipeline
        pipeline_cfg = dataset_config.get("pipeline", [])
        self.transform_pipeline = self._build_pipeline(pipeline_cfg)

        # Keys (video IDs) for indexing
        self.video_keys = list(self.video_items.keys())

    def _scan_videos(self):
        """
        Scan video folders in all roots and return their intersection as a dictionary.

        Structure:
            folder_paths[key] / data_prefix[key] / <video_name> / frame_*.png

        Returns:
            {
                "<video_name>": {
                    "<key>_path": [frame_path_1, frame_path_2, ...],
                    ...
                },
                ...
            }
        """
        extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        # key -> {video_name -> [frames]}
        frames_by_key = {}

        for key, root in self.folder_paths.items():
            root = Path(root)
            path_prefix = root / self.data_prefix.get(key, "")

            if not path_prefix.exists():
                raise ValueError(f"Folder {path_prefix} does not exist")
            if not path_prefix.is_dir():
                raise ValueError(f"{path_prefix} is not a directory")

            video_to_frames = {}

            # Treat each immediate subdirectory as a video
            for subdir in sorted(d for d in path_prefix.iterdir() if d.is_dir()):
                frame_paths = []
                for ext in extensions:
                    frame_paths.extend(subdir.glob(f"*{ext}"))

                # Only keep subdirs that actually contain frames
                if frame_paths:
                    # Sort frames by name for consistent temporal order
                    frame_paths = sorted(frame_paths, key=lambda p: p.name)
                    video_to_frames[subdir.name] = [str(p) for p in frame_paths]

            if not video_to_frames:
                raise ValueError(
                    f"No video folders (subdirs with image frames) found in {path_prefix}"
                )

            frames_by_key[key] = video_to_frames
            print(f"Found {len(video_to_frames)} video folders in {path_prefix}")

        # Intersect video names across keys
        all_keys = list(frames_by_key.keys())
        common_video_names = set(frames_by_key[all_keys[0]].keys())
        for key in all_keys[1:]:
            common_video_names &= set(frames_by_key[key].keys())

        if not common_video_names:
            raise ValueError("No common video folders found across all folder_paths")

        # Build final mapping
        result = {}
        for i, video_name in enumerate(sorted(common_video_names)):
            sample_dict = {}
            for key in self.folder_paths:
                # Keep naming convention: "<key>_path"
                # but value is now a *list* of frame file paths (one video).
                sample_dict[f"{key}_path"] = frames_by_key[key][video_name]

            result[video_name] = sample_dict

            if self.max_dataset_size is not None and i + 1 >= self.max_dataset_size:
                break

        print(f"Found {len(result)} common videos across all folders")
        return result

    def _build_pipeline(self, pipeline_cfg):
        """Build the data processing pipeline using getattr for dynamic class loading."""
        transforms_list = []

        for transform_cfg in pipeline_cfg:
            transform_cfg = transform_cfg.copy()  # avoid modifying original
            transform_type = transform_cfg.pop("type")

            transform_class = getattr(transforms, transform_type)
            transform = transform_class(**transform_cfg)
            transforms_list.append(transform)

        return transforms_list

    def __len__(self):
        if self.max_dataset_size is not None:
            return min(len(self.video_items), self.max_dataset_size) * self.inflate_factor
        return len(self.video_items) * self.inflate_factor

    def __getitem__(self, idx):

        idx = idx % len(self.video_items)
        video_key = self.video_keys[idx]

        # Shallow copy so transforms can mutate in-place if they want
        data = dict(self.video_items[video_key])

        for transform in self.transform_pipeline:
            data = transform(data)

        return data

class MetaGuidedVideoDataset(BasicVideoDataset):
    """BasicVideoDataset + meta-file guided filtering/ordering + optional frame-count checks."""

    def __init__(self, **dataset_config):
        # meta config (stored before super().__init__ because super calls self._scan_videos())
        self.meta_info_file = dataset_config.get("meta_info_file", None)
        self.meta_strict = bool(dataset_config.get("meta_strict", True))
        self.check_num_frames = bool(dataset_config.get("check_num_frames", True))

        self.meta_video_order, self.meta_expected_frames = self._load_meta(self.meta_info_file)

        super().__init__(**dataset_config)

    def _load_meta(self, meta_info_file):
        """Return (ordered_video_list, expected_frame_count_dict). 3rd column ignored."""
        if not meta_info_file:
            return None, {}

        meta_path = Path(meta_info_file)
        if not meta_path.exists() or not meta_path.is_file():
            raise ValueError(f"meta_info_file does not exist or is not a file: {meta_path}")

        order = []
        expected = {}

        with meta_path.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                vid = parts[0]
                try:
                    n_frames = int(parts[1])
                except ValueError as e:
                    raise ValueError(f"Invalid frame count in meta at line {ln}: {line}") from e

                order.append(vid)
                expected[vid] = n_frames

        if not order:
            raise ValueError(f"Meta file is empty/unreadable: {meta_path}")

        return order, expected

    def _scan_videos(self):
        """
        Reuse BasicVideoDataset scanning, then meta-filter/reorder/check.
        """
        # Get full intersection first (apply max_dataset_size AFTER meta ordering)
        orig_max = self.max_dataset_size
        self.max_dataset_size = None
        all_common = super()._scan_videos()
        self.max_dataset_size = orig_max

        # No meta => behave exactly like base (but respecting original max_dataset_size)
        if self.meta_video_order is None:
            if orig_max is not None:
                keys = list(all_common.keys())[:orig_max]
                return {k: all_common[k] for k in keys}
            return all_common

        # Meta-filter + order
        missing = [v for v in self.meta_video_order if v not in all_common]
        if missing and self.meta_strict:
            raise ValueError(
                f"{len(missing)} videos listed in meta are missing from the paired dataset intersection. "
                f"Example missing: {missing[:10]}"
            )

        ordered_names = [v for v in self.meta_video_order if v in all_common]

        # Optional frame-count checks (2nd column)
        if self.check_num_frames:
            mismatches = []
            for vid in ordered_names:
                exp = self.meta_expected_frames.get(vid)
                if exp is None:
                    continue
                sample = all_common[vid]  # {"<key>_path": [..frames..], ...}
                for k in self.folder_paths.keys():
                    frame_list = sample.get(f"{k}_path", [])
                    if len(frame_list) != exp:
                        mismatches.append((vid, k, exp, len(frame_list)))
            if mismatches:
                raise ValueError(
                    "Frame count mismatches (vid, key, expected, found), examples: "
                    f"{mismatches[:10]}"
                )

        # Apply max_dataset_size after meta ordering
        if orig_max is not None:
            ordered_names = ordered_names[:orig_max]

        return {vid: all_common[vid] for vid in ordered_names}
