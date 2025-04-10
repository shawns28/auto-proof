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
    mat_version_start = data_config['client']['mat_version_start']
    mat_version_end = data_config['client']['mat_version_end']
    roots_dir = f'{data_config['data_dir']}roots_{mat_version_start}_{mat_version_end}/'
    roots = data_utils.load_txt(f'{roots_dir}{data_config['raw_edits']['post_raw_edit_roots']}')
    print("roots len", len(roots))

    mat_versions = data_config['proofread']['mat_versions']
    mat_dict = {}
    for mat_version in mat_versions:
        mat_dict[mat_version] = proofread_utils.convert_proofread_csv_to_txt(data_config, mat_version)
        print("mat version", mat_version, "len", len(mat_dict[mat_version]))
    
    combined_proofread_roots = proofread_utils.combine_proofread_roots(mat_dict)
    print("combined roots len", len(combined_proofread_roots))

    roots_set = set(roots)
    changed_proofread_roots = []
    for proofread_root in combined_proofread_roots:
        if proofread_root in roots_set:
            changed_proofread_roots.append(proofread_root)
    print("changed proofread len", len(changed_proofread_roots))
    
    combined_proofread_roots = [root for root in combined_proofread_roots if root not in changed_proofread_roots]
    print("combined roots after removing changed roots len", len(combined_proofread_roots))
    
    mat_version1 = mat_versions[0]
    unique_proofread_roots1 = [root for root in combined_proofread_roots if root in mat_dict[mat_version1]]
    print("len of unique_proofread_roots1", len(unique_proofread_roots1))
    data_utils.save_txt(f'{data_config['data_dir']}{data_config['proofread']['proofread_dir']}{mat_version1}_unique.txt', unique_proofread_roots1)

    #NOTE: Change for more than 2 mat versions and if mat versions aren't 943 and 1300
    mat_version2 = ""
    if len(mat_versions) == 2:
        mat_version2 = mat_versions[1]
        unique_proofread_roots2 = [root for root in combined_proofread_roots if (root in mat_dict[mat_version2] and root not in unique_proofread_roots1)]
        print("len of unique_proofread_roots2", len(unique_proofread_roots2))
        data_utils.save_txt(f'{data_config['data_dir']}{data_config['proofread']['proofread_dir']}{mat_version2}_unique.txt', unique_proofread_roots2)
        assert len(unique_proofread_roots2) + len(unique_proofread_roots1) == len(combined_proofread_roots)

    # TODO: Uncomment below to actually make copies
    copied_roots = proofread_utils.make_copies_of_roots(combined_proofread_roots, data_config['proofread']['copy_count'])
    # copied_roots = data_utils.add_identifier_to_roots(combined_proofread_roots)
    print("copied roots len", len(copied_roots))
    roots = data_utils.add_identifier_to_roots(roots)
    roots.extend(copied_roots)
    print("roots final len", len(roots))

    data_utils.save_txt(f'{data_config['data_dir']}{data_config['proofread']['proofread_dir']}{mat_version1}_{mat_version2}_changed.txt', changed_proofread_roots)
    data_utils.save_txt(f'{data_config['data_dir']}{data_config['proofread']['proofread_dir']}{mat_version1}_{mat_version2}.txt', combined_proofread_roots)
    data_utils.save_txt(f'{roots_dir}{data_config['proofread']['post_proofread_roots']}', roots)

if __name__ == "__main__":
    data_config = data_utils.get_data_config()
    main(data_config)