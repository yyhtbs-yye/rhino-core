import torch

class PairedCrop:
    def __init__(self, crop_size, is_pad_zeros=True, random_crop=False,
                 scale_factor_dict=None):
        """
        Args:
            crop_size: int or [h, w] in the reference resolution (scale_factor == 1).
            is_pad_zeros: whether to pad to full crop size if near borders.
            random_crop: if False -> center crop, if True -> random crop.
            scale_factor_dict: dict like {'lq': 1, 'gt': 4, ...}
                               keys must exist in `results` at __call__ time.
        """
        if scale_factor_dict is None:
            scale_factor_dict = {}

        if not scale_factor_dict:
            raise ValueError("scale_factor_dict must not be empty.")

        # crop_size is defined in the resolution of the reference key (scale==1)
        if isinstance(crop_size, (list, tuple)):
            self.crop_size = list(crop_size)
        else:
            self.crop_size = [crop_size, crop_size]

        self.is_pad_zeros = is_pad_zeros
        self.random_crop = random_crop
        self.scale_factor_dict = scale_factor_dict  # e.g. {'lq': 1, 'gt': 4}

        # Require one reference key with scale 1
        ref_keys = [k for k, v in self.scale_factor_dict.items() if v == 1]
        if not ref_keys:
            raise ValueError(
                "scale_factor_dict must contain at least one key with "
                "scale_factor == 1 to serve as reference resolution."
            )
        self.ref_key = ref_keys[0]  # use the first one as the reference

    def __call__(self, results):
        # 1) Check that all required keys are present in results
        missing = [k for k in self.scale_factor_dict.keys() if k not in results]
        if missing:
            raise KeyError(
                f"The following keys from scale_factor_dict are missing in results: {missing}"
            )

        # 2) Get reference image and its spatial size (last two dims)
        ref_img = results[self.ref_key]
        if not torch.is_tensor(ref_img):
            raise TypeError(
                f"results['{self.ref_key}'] must be a torch.Tensor, got {type(ref_img)}"
            )
        if ref_img.dim() < 2:
            raise ValueError(
                f"results['{self.ref_key}'] must have at least 2 dimensions, "
                f"got shape {tuple(ref_img.shape)}"
            )

        h_ref, w_ref = ref_img.shape[-2:]
        ref_crop_h, ref_crop_w = self.crop_size

        # 3) Compute reference crop offsets (in reference resolution)
        if self.random_crop:
            # Sample offsets in reference spatial coordinates
            max_x = max(0, w_ref - ref_crop_w)
            max_y = max(0, h_ref - ref_crop_h)
            ref_x_offset = torch.randint(0, max_x + 1, (1,)).item()
            ref_y_offset = torch.randint(0, max_y + 1, (1,)).item()
        else:
            # Center crop in reference resolution
            ref_x_offset = max(0, (w_ref - ref_crop_w) // 2)
            ref_y_offset = max(0, (h_ref - ref_crop_h) // 2)

        # 4) Process each key according to its scale factor
        for key, declared_scale in self.scale_factor_dict.items():
            img = results[key]
            if not torch.is_tensor(img):
                raise TypeError(
                    f"results['{key}'] must be a torch.Tensor, got {type(img)}"
                )
            if img.dim() < 2:
                raise ValueError(
                    f"results['{key}'] must have at least 2 dimensions, "
                    f"got shape {tuple(img.shape)}"
                )

            h, w = img.shape[-2:]

            # 4a) Check compatibility of shape with reference and declared scale
            if h_ref == 0 or w_ref == 0:
                raise ValueError(
                    f"Reference image '{self.ref_key}' has zero spatial dimension: "
                    f"({h_ref}, {w_ref})"
                )

            if h % h_ref != 0 or w % w_ref != 0:
                raise ValueError(
                    f"Incompatible spatial size for key '{key}': "
                    f"({h}, {w}) is not an integer multiple of the reference "
                    f"({h_ref}, {w_ref})."
                )

            scale_h = h // h_ref
            scale_w = w // w_ref
            if scale_h != scale_w:
                raise ValueError(
                    f"Non-isotropic scaling for key '{key}': "
                    f"scale_h={scale_h}, scale_w={scale_w}."
                )

            inferred_scale = scale_h
            if declared_scale != inferred_scale:
                raise ValueError(
                    f"Declared scale_factor_dict['{key}'] = {declared_scale} "
                    f"does not match inferred scale = {inferred_scale} from shapes."
                )

            scale = inferred_scale  # use the checked scale

            # 4b) Compute crop size in this key's resolution
            key_crop_h = ref_crop_h * scale
            key_crop_w = ref_crop_w * scale

            # 4c) Compute offsets for this key
            if self.random_crop:
                # Use the same reference offsets, scaled
                x_offset = ref_x_offset * scale
                y_offset = ref_y_offset * scale
            else:
                # Center crop per key (still spatially aligned by symmetry)
                x_offset = max(0, (w - key_crop_w) // 2)
                y_offset = max(0, (h - key_crop_h) // 2)

            # 4d) Crop on the last two dimensions only
            y1, y2 = y_offset, min(y_offset + key_crop_h, h)
            x1, x2 = x_offset, min(x_offset + key_crop_w, w)
            cropped = img[..., y1:y2, x1:x2]

            # 4e) Optional zero padding (bottom/right) on spatial dims only
            cur_h, cur_w = cropped.shape[-2:]
            if self.is_pad_zeros and (cur_h < key_crop_h or cur_w < key_crop_w):
                pad_h = key_crop_h - cur_h
                pad_w = key_crop_w - cur_w

                # Create a new zero tensor with the target spatial size
                new_shape = (*cropped.shape[:-2], key_crop_h, key_crop_w)
                padded = cropped.new_zeros(new_shape)
                padded[..., :cur_h, :cur_w] = cropped
                cropped = padded

            results[key] = cropped

        # keys in `results` that are not in scale_factor_dict are left untouched
        return results
