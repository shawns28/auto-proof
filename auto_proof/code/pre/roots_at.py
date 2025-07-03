from auto_proof.code.pre import data_utils

import numpy as np
import h5py
import time
from cloudvolume import CloudVolume
import time
from typing import List, Tuple, Any, Optional # Import for type hinting

def get_roots_at(vertices: np.ndarray, cv_seg: CloudVolume, resolution: np.ndarray) -> Tuple[bool, Optional[Exception], Optional[np.ndarray]]:
    """Retrieves the root IDs at specified vertex coordinates from CloudVolume at specified segmentation.

    This function queries a CloudVolume segmentation for the root id at each
    given 3D coordinate. It includes retry logic to handle potential transient
    network errors or API issues. The input vertices are expected in their
    original scale and are converted to the CloudVolume's internal resolution
    before querying.

    Args:
        vertices: A NumPy array of shape `(N, 3)` representing the 3D coordinates
            of N points. These coordinates are assumed to be in a specific scale
            and will be downscaled by `resolution`.
        cv_seg: An initialized `CloudVolume` instance pointing to the segmentation
            layer.
        resolution: A NumPy array of shape `(3,)` representing the resolution
            (e.g., voxel size in nanometers) in x, y, and z dimensions. Used
            to convert `vertices` to CloudVolume's internal coordinate system.

    Returns:
        A tuple:
        - bool: True if the root IDs were successfully retrieved for all vertices,
            False otherwise.
        - Optional[Exception]: An Exception object if an error occurred after all
            retries, None otherwise.
        - Optional[np.ndarray]: A NumPy array of shape `(N,)` containing the
            segmentation ID (root ID) at each input vertex, or None if the
            operation failed.
    """
    vertices = vertices / resolution
    input_vertices = [(vertices[i][0].item(), vertices[i][1].item(), vertices[i][2].item()) for i in range(len(vertices))]
    retries = 3
    delay = 5
    for attempt in range(0, retries + 1):   
        try:
            root_at_dict = cv_seg.scattered_points(input_vertices)
            root_at_arr = np.array([root_at_dict[input_vertices[i]] for i in range(len(vertices))])
            return True, None, root_at_arr
        except Exception as e:
            if attempt < retries:
                time.sleep(delay)
                continue
            else:
                return False, e, None
    return False, None, None # Shouldn't enter
