import torch

def list_of_dict_aggr_to_dict(list_of_dict):
    """
    """
    if not list_of_dict:
        return {}

    keys = set().union(*(d.keys() for d in list_of_dict))
    out = {}

    for k in keys:
        vals = []
        for d in list_of_dict:
            if k not in d or d[k] is None:
                continue
            v = d[k]
            if torch.is_tensor(v):
                v = v.detach()
                v = v.mean() if v.ndim > 0 else v
            else:
                v = v
            vals.append(v)
        if vals:
            out[k] = sum(vals) / len(vals)

    return out
