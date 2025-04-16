from auto_proof.code.pre import data_utils

import numpy as np
import h5py
import time
from cloudvolume import CloudVolume
import time

def get_roots_at(feature_path, cv_seg, resolution):
    with h5py.File(feature_path, 'r') as f:
        vertices = f['vertices'][:]
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
