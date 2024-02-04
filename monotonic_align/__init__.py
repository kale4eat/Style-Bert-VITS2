from numpy import zeros, int32, float32
import torch
from torch import from_numpy

from .core import maximum_path_jit


def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor):
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = neg_cent.data.cpu().numpy().astype(float32)
    path = zeros(neg_cent.shape, dtype=int32)

    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(int32)
    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(int32)
    maximum_path_jit(path, neg_cent, t_t_max, t_s_max)
    return from_numpy(path).to(device=device, dtype=dtype)


def maximum_path_pytorch(paths, values, t_ys, t_xs):
    b = paths.shape[0]
    max_neg_val: float = -1e9
    for i in range(b):
        path = paths[i]
        value = values[i]
        t_y = int(t_ys[i].item())
        t_x = int(t_xs[i].item())

        for y in range(t_y):
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                v_cur = max_neg_val if x == y else float(value[y - 1, x].item())
                v_prev = (
                    0.0
                    if y == 0
                    else max_neg_val
                    if x == 0
                    else float(value[y - 1, x - 1].item())
                )
                value[y, x] += max(v_prev, v_cur)

        index = t_x - 1
        for y in range(t_y - 1, -1, -1):
            path[y, index] = 1
            if index != 0 and (
                index == y or value[y - 1, index] < value[y - 1, index - 1]
            ):
                index -= 1


def maximum_path_new(neg_cent: torch.Tensor, mask: torch.Tensor):
    device = neg_cent.device
    dtype = neg_cent.dtype

    # PyTorchのテンソル操作を使用してpathを初期化
    path = torch.zeros(neg_cent.shape, dtype=torch.int32, device=device)

    # maskを使用してt_t_maxとt_s_maxを計算
    t_t_max = mask.sum(dim=1)[:, 0].to(dtype=torch.int32)
    t_s_max = mask.sum(dim=2)[:, 0].to(dtype=torch.int32)

    maximum_path_pytorch(path, neg_cent, t_t_max, t_s_max)

    return path.type(dtype)
