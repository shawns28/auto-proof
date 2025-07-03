from auto_proof.code.pre import data_utils

import pandas as pd
import numpy as np

def convert_proofread_csv_to_txt(mat_version: int, proofread_csv: str) -> list[str]:
    """Converts a proofread CSV file to a list of root IDs.

    Args:
        mat_version: The materialization version number. This affects how
            axon statuses are filtered.
        proofread_csv: The file path to the proofread CSV.

    Returns:
        A list of root IDs (as strings) that meet the specified axon status
        criteria for the given materialization version.
    """
    df = pd.read_csv(proofread_csv)
    if mat_version <= 943:
        filtered_df = df[df['status_axon'] != 'non']
    else:
        filtered_df = df[df['status_axon'] == 't']
    root_ids = filtered_df['root_id']
    root_ids = [str(root) for root in root_ids]
    return root_ids

def combine_proofread_roots(mat_dict: dict[int, list[str]]) -> list[str]:
    """Combines root IDs from multiple materialization versions into a single,
    unique, inclusive list.

    Args:
        mat_dict: A dictionary where keys are materialization versions (int)
            and values are lists of proofread root IDs (str) for that version.

    Returns:
        A list of unique root IDs, combining all roots from all provided
        materialization versions.
    """
    combined_set = set()
    for mat_version in mat_dict:
        combined_set.update(mat_dict[mat_version])
    return list(combined_set)

def make_copies_of_roots(roots: list[str], copy_count: int) -> list[str]:
    """Makes copies of each root ID and appends a unique identifier to each copy.

    For example, if `roots` contains ['123'] and `copy_count` is 2, the result
    will be ['123_000', '123_001']. The identifier format adjusts for counts
    greater than or equal to 10.

    Args:
        roots: A list of proofread root IDs (strings).
        copy_count: The total number of copies to create for each root.
            The maximum supported copy count is 99 (due to identifier formatting).

    Returns:
        A new list containing copies of the original roots, each with a unique
        identifier appended (e.g., '_000', '_001', '_010').
    """
    new_roots = []
    for root in roots:
        for i in range(min(copy_count, 10)):  # Handles _000 to _009
            new_str = str(root) + '_00' + str(i)
            new_roots.append(new_str)
        if copy_count > 10:  # Handles _010 onwards
            for i in range(10, copy_count):
                new_str = str(root) + '_0' + str(i)
                new_roots.append(new_str)
    return new_roots