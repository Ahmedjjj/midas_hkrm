import numpy as np


def map_disp_to_0_1(disp: np.ndarray) -> np.ndarray:
    """
    Map the disparity values into the 0-1 range.

    Args:
        disp (np.ndarray): original disparity map

    Returns:
        np.ndarray: mapped disparity map
    """

    res = disp.copy().astype(float)
    min_value, max_value = np.min(res), np.max(res)
    if min_value != max_value:
        return (res - min_value) / (max_value - min_value)
    elif max_value == 0:
        return res
    else:
        return res / max_value


def map_depth_to_disp(depth: np.ndarray) -> np.ndarray:
    """
    Map a depth map to a disparity map, ignoring zero values in the original depth map.

    Args:
        depth (np.ndarray): depth map, 0 values correspond to invalid pixels

    Returns:
        np.ndarray: depth map
    """
    mask = depth != 0
    disparity = depth.copy().astype(float)
    disparity[mask] = 1.0 / disparity[mask]
    return map_disp_to_0_1(disparity)
