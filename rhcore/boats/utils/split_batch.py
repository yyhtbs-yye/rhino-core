import torch

def split_batch(x, parts):
    
    # returns a list of length `parts` mirroring x's structure
    if torch.is_tensor(x):
        # torch.chunk handles non-divisible sizes
        return list(torch.chunk(x, parts, dim=0))
    if isinstance(x, dict):
        per = [dict() for _ in range(parts)]
        for k, v in x.items():
            chunks = split_batch(v, parts)
            for i in range(parts):
                per[i][k] = chunks[i]
        return per
    if isinstance(x, (list, tuple)):
        elems = [split_batch(v, parts) for v in x]
        out = []
        for i in range(parts):
            out.append(type(x)(chunks[i] for chunks in elems))
        return out
    # non-tensor leaf: replicate reference (ok for e.g. scalars/strings)
    return [x for _ in range(parts)]
