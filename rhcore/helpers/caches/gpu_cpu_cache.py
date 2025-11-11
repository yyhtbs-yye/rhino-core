from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class GpuCpuFilterCache:
    """
    Two-level LRU cache:
      - GPU cache: up to `max_gpu` filters
      - CPU cache: up to `max_cpu` filters

    When the GPU cache exceeds `max_gpu`, the least-recently-used entry
    is offloaded to CPU. When the CPU cache exceeds `max_cpu`, the
    least-recently-used entry is dropped.
    """

    def __init__(self, max_gpu: int = 128, max_cpu: int = 1024):
        self.max_gpu = max_gpu
        self.max_cpu = max_cpu
        self.gpu_cache = OrderedDict()  # key -> Tensor on GPU / accelerator
        self.cpu_cache = OrderedDict()  # key -> Tensor on CPU

    @staticmethod
    def _touch(cache: OrderedDict, key):
        """Mark `key` as recently used and return its value."""
        value = cache.pop(key)
        cache[key] = value
        return value

    def _add_to_cpu(self, key, tensor: torch.Tensor):
        """Insert tensor (CPU) into CPU cache (with LRU eviction)."""
        if tensor.device.type != "cpu":
            tensor = tensor.to("cpu")
        self.cpu_cache[key] = tensor
        # Enforce CPU capacity (drop LRU)
        while len(self.cpu_cache) > self.max_cpu:
            self.cpu_cache.popitem(last=False)

    def _add_to_gpu(self, key, tensor: torch.Tensor, device: torch.device):
        """Insert tensor into GPU cache, offloading LRU to CPU if needed."""
        if tensor.device != device:
            tensor = tensor.to(device)
        self.gpu_cache[key] = tensor
        # Enforce GPU capacity: offload LRU to CPU
        while len(self.gpu_cache) > self.max_gpu:
            old_key, old_tensor = self.gpu_cache.popitem(last=False)
            self._add_to_cpu(old_key, old_tensor)

    def get(self, key, device: torch.device, create_fn):
        """
        Retrieve filter for `key` on `device`.

        Args:
            key: hashable identifier of the filter.
            device: torch.device where the returned tensor should live.
            create_fn: callable(device) -> Tensor created on the given device.
        """
        # CPU request
        if device.type == "cpu":
            if key in self.cpu_cache:
                return self._touch(self.cpu_cache, key)

            if key in self.gpu_cache:
                # Move from GPU to CPU on demand
                tensor_gpu = self._touch(self.gpu_cache, key)
                tensor_cpu = tensor_gpu.to("cpu")
                self._add_to_cpu(key, tensor_cpu)
                return tensor_cpu

            # Not cached anywhere: create directly on CPU
            tensor = create_fn(device)
            self._add_to_cpu(key, tensor)
            return tensor

        # Accelerator / GPU request
        if key in self.gpu_cache:
            tensor = self._touch(self.gpu_cache, key)
            if tensor.device != device:
                tensor = tensor.to(device)
                self.gpu_cache[key] = tensor
            return tensor

        if key in self.cpu_cache:
            tensor_cpu = self._touch(self.cpu_cache, key)
            self._add_to_gpu(key, tensor_cpu, device)
            return self.gpu_cache[key]

        # Not cached: create on requested device
        tensor = create_fn(device)
        self._add_to_gpu(key, tensor, device)
        return tensor