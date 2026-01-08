import torch

class RandomTimeChunk:
    def __init__(self, keys, num_frames=15):
        
        self.keys = keys
        self.num_frames = num_frames
        
    def __call__(self, results):

        total_frames = results[self.keys[0]].size(0)

        start_idx = torch.randint(0, total_frames - self.num_frames + 1, ())

        for key in self.keys:
            video = results[key]
            results[key] = video[start_idx : start_idx + self.num_frames]
        
        return results