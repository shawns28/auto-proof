from auto_proof.code.pre import data_utils

import pandas as pd
import numpy as np

def convert_proofread_csv_to_txt(data_config, mat_version):
    """Converts proofread csv to txt.
    
    Args:
        data_config
        mat_version: Identifier for which proofread csv to convert
    Returns:
        roots converted
    """
    proofread_csv = data_config['proofread'][f'{mat_version}_csv']
    df = pd.read_csv(proofread_csv)
    filtered_df = df[df['status_axon'] != 'non']
    root_ids = filtered_df['root_id']
    root_ids = [str(root) for root in root_ids]
    root_ids_array = np.array(root_ids)
    proofread_root_path = f'{data_config['data_dir']}{data_config['proofread']['proofread_dir']}{mat_version}.txt'
    data_utils.save_txt(proofread_root_path, root_ids_array)
    return root_ids

def combine_proofread_roots(mat_dict):
    """Combines the roots from each mat version into one unique inclusive list.
    
    Args:
        mat_dict: (mat version: proofread roots at mat version)
    Returns:
        list of combined inclusive roots
    """
    combined_set = set()
    for mat_version in mat_dict:
        combined_set.update(mat_dict[mat_version])
    return list(combined_set)

def make_copies_of_roots(roots, copy_count):
    """Makes copies of each root and adds identifier.

    Args:
        roots: proofread roots
        copy_count: Number of copies to have total < 100
    Returns:
        roots with copies and identifiers for each copy index
    """
    new_roots = []
    for root in roots:
        for i in range(min(copy_count, 10)):
            new_str = str(root) + '_00' + str(i)
            new_roots.append(new_str)
        if copy_count > 10:
            for i in range(10, copy_count):
                new_str = str(root) + '_0' + str(i)
                new_roots.append(new_str)
    return new_roots