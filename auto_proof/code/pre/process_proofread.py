from auto_proof.code.pre import data_utils
from auto_proof.code.pre import proofread_utils

import os
import numpy as np
from tqdm import tqdm
import h5py
import multiprocessing
import glob

def main(data_config):
    """
        TODO: Fill in
    """
    roots = data_utils.load_txt(data_config['proofread']['pre_root_path'])
    roots = list(data_utils.add_identifier_to_roots(roots))
    print("roots len", len(roots))

    mat_versions = data_config['proofread']['mat_versions']
    mat_dict = {}
    for mat_version in mat_versions:
        mat_dict[mat_version] = proofread_utils.convert_proofread_csv_to_txt(data_config, mat_version)
        print("mat version", mat_version, "len", len(mat_dict[mat_version]))
    
    combined_roots = proofread_utils.combine_proofread_roots(mat_dict)
    print("combined roots len", len(combined_roots))
    
    copied_roots = proofread_utils.make_copies_of_roots(combined_roots, data_config['proofread']['copy_count'])
    print("copied roots len", len(copied_roots))

    roots_set = set(roots)
    changed_proofread_roots = set()
    for copied_root in copied_roots:
        transformed_root = copied_root[:-4] + '_000'
        if transformed_root in roots_set:
            changed_proofread_roots.add(transformed_root)
        else:
            roots.append(copied_root)
    print("changed proofread len", len(changed_proofread_roots))
    
    combined_roots = [root for root in combined_roots if (str(root) + '_000') not in changed_proofread_roots]
    print("combined roots after removing changed roots len", len(combined_roots))
    print("roots final len", len(roots))

    data_utils.save_txt(data_config['proofread']['combined_path'], combined_roots)
    data_utils.save_txt(data_config['proofread']['post_root_path'], roots)

if __name__ == "__main__":
    data_config = data_utils.get_data_config()
    main(data_config)