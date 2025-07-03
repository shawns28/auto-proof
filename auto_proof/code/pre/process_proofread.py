from auto_proof.code.pre import data_utils
from auto_proof.code.pre import proofread_utils

import time
import os

def main():
    """Processes proofread roots across different materialization versions.

    This function performs the following steps:
    1. Loads configuration for data and client settings.
    2. Retrieves initial raw edit roots.
    3. Converts proofread CSVs from specified materialization versions into
       text files of root IDs and stores them in a dictionary.
    4. Combines all proofread roots into a single unique list.
    5. Identifies proofread roots at earlier mat versions that have undergone
       further changes and removes them from the proofread list.
    7. Creates copies of these proofread roots, appending unique identifiers.
    8. Saves various intermediate and final lists of roots to text files for
       further use in the auto-proof pipeline.
    """
    data_config = data_utils.get_config('data')
    client_config = data_utils.get_config('client')
    mat_version_start = client_config['client']['mat_version_start']
    mat_version_end = client_config['client']['mat_version_end']
    roots_dir = f'{data_config['data_dir']}roots_{mat_version_start}_{mat_version_end}/'
    roots = data_utils.load_txt(f'{roots_dir}{data_config['raw_edits']['post_raw_edit_roots']}')
    print("Initial raw edit roots count:", len(roots))

    proofread_dir = f'{data_config['data_dir']}{data_config['proofread']['proofread_dir']}'
    os.makedirs(proofread_dir, exist_ok=True)

    mat_versions = data_config['proofread']['mat_versions']
    mat_dict = {}
    for mat_version in mat_versions:
        proofread_csv = data_config['proofread'][f'csv_{mat_version}']
        proofread_root_path = f'{data_config['data_dir']}{data_config['proofread']['proofread_dir']}{mat_version}.txt'
        mat_dict[mat_version] = proofread_utils.convert_proofread_csv_to_txt(mat_version, proofread_csv)
        data_utils.save_txt(proofread_root_path, mat_dict[mat_version])
        print(f"Proofread roots for materialization version {mat_version}: {len(mat_dict[mat_version])}")
    
    combined_proofread_roots = proofread_utils.combine_proofread_roots(mat_dict)
    print("Combined unique proofread roots count:", len(combined_proofread_roots))

    roots_set = set(roots)
    changed_proofread_roots = []
    # Identify roots that were initially present and also proofread
    for proofread_root in combined_proofread_roots:
        if proofread_root in roots_set:
            changed_proofread_roots.append(proofread_root)
    print("Changed proofread roots (present in initial roots and proofread) count:", len(changed_proofread_roots))
    
    # Filter out the changed roots from the combined proofread roots
    combined_proofread_roots = [root for root in combined_proofread_roots if root not in changed_proofread_roots]
    print("Combined proofread roots (after removing changed roots) count:", len(combined_proofread_roots))
    
    # Process roots unique to the first materialization version
    mat_version1 = mat_versions[0]
    unique_proofread_roots1 = [root for root in combined_proofread_roots if root in mat_dict[mat_version1]]
    print(f"Unique proofread roots for version {mat_version1}: {len(unique_proofread_roots1)}")
    data_utils.save_txt(f'{data_config['data_dir']}{data_config['proofread']['proofread_dir']}{mat_version1}_unique.txt', unique_proofread_roots1)
    unique_proofread_roots1_copied = proofread_utils.make_copies_of_roots(unique_proofread_roots1, data_config['proofread']['copy_count'])
    data_utils.save_txt(f'{data_config['data_dir']}{data_config['proofread']['proofread_dir']}{mat_version1}_unique_copied.txt', unique_proofread_roots1_copied)

    # Process roots unique to the second materialization version and not in the first
    mat_version2 = mat_versions[1]
    unique_proofread_roots2 = [root for root in combined_proofread_roots if (root in mat_dict[mat_version2] and root not in unique_proofread_roots1)]
    print(f"Unique proofread roots for version {mat_version2} (not in {mat_version1}): {len(unique_proofread_roots2)}")
    data_utils.save_txt(f'{data_config['data_dir']}{data_config['proofread']['proofread_dir']}{mat_version2}_unique.txt', unique_proofread_roots2)
    unique_proofread_roots2_copied = proofread_utils.make_copies_of_roots(unique_proofread_roots2, data_config['proofread']['copy_count'])
    data_utils.save_txt(f'{data_config['data_dir']}{data_config['proofread']['proofread_dir']}{mat_version2}_unique_copied.txt', unique_proofread_roots2_copied)

    # Assert that the sum of unique roots from both versions equals the total combined proofread roots
    assert len(unique_proofread_roots2) + len(unique_proofread_roots1) == len(combined_proofread_roots)

    # Create copies of all combined proofread roots
    copied_roots = proofread_utils.make_copies_of_roots(combined_proofread_roots, data_config['proofread']['copy_count'])
    print("Copied combined proofread roots count:", len(copied_roots))

    # Add identifiers and extend with copied proofread roots
    roots = data_utils.add_identifier_to_roots(roots)
    roots.extend(copied_roots)
    print("Final combined roots list count (raw edits + copied proofread roots):", len(roots))

    # Save final lists to text files
    data_utils.save_txt(f'{data_config['data_dir']}{data_config['proofread']['proofread_dir']}{mat_version1}_{mat_version2}_copied.txt', copied_roots)
    data_utils.save_txt(f'{data_config['data_dir']}{data_config['proofread']['proofread_dir']}{mat_version1}_{mat_version2}_changed.txt', changed_proofread_roots)
    data_utils.save_txt(f'{data_config['data_dir']}{data_config['proofread']['proofread_dir']}{mat_version1}_{mat_version2}.txt', combined_proofread_roots)
    data_utils.save_txt(f'{roots_dir}{data_config['proofread']['post_proofread_roots']}', roots)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time # Calculate elapsed time
    print(f"\nProcess completed in {elapsed_time:.2f} seconds.")