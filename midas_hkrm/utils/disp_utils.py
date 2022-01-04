import numpy as np


def map_disp_to_0_1(disp):
    res = disp.copy().astype(float)
    min_value, max_value = np.min(res), np.max(res)
    if min_value != max_value:
        return (res - min_value) / (max_value - min_value)
    elif max_value == 0:
        return res
    else:
        return res / max_value


def map_depth_to_disp(depth):
    mask = depth != 0
    disparity = depth.copy().astype(float)
    disparity[mask] = 1.0 / disparity[mask]
    return map_disp_to_0_1(disparity)
