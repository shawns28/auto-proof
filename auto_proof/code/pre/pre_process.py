from auto_proof.code.pre import data_utils

from caveclient import CAVEclient
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def main():
    _, _, mat_version, client, data_directory = data_utils.initialize()
    # want to test if this works while reatining the working data directory
    # data_directory = '../../testing_data'

    og_df_path = f'{data_directory}/240927/minnie_splitlog_240927.feather'
    og_edit_path = f'{data_directory}/240927/minnie_splitlog_240927.npy'

    if not (os.path.exists(og_df_path) and os.path.exists(og_edit_path)):
        print("Error: initial starting df and edits doesn't exist")

    pruned_df_path = f'{data_directory}/splitlog_{mat_version}.feather'    
    if not (os.path.exists(pruned_df_path)):
        print("saving pruned df")
         # [date, timestamp, mean_coord_x, mean_coord_y, mean_coord_z, operation_id]
        og_df = pd.read_feather(og_df_path)
        data_utils.save_pruned_df(client, mat_version, og_df, pruned_df_path)
        print("done saving pruned df")
    
    pruned_edit_path = f'{data_directory}/splitlog_{mat_version}.npy'
    if not (os.path.exists(pruned_edit_path)):
        print("saving pruned edits")
        # [operation_id, sink supervoxel id, source_supervoxel id]
        og_edits = np.load(og_edit_path)
        pruned_df = pd.read_feather(pruned_df_path)
        data_utils.save_pruned_edits(og_edits, pruned_df, pruned_edit_path)
        print("done saving pruned edits")

    op_to_svs_path = f'{data_directory}/operation_to_supervoxels_{mat_version}.pkl'
    if not (os.path.exists(op_to_svs_path)):
        print("saving op_to_svs")
        pruned_edits = np.load(pruned_edit_path)
        data_utils.save_operation_to_svs(pruned_edits, op_to_svs_path)
        print("done saving op_to_svs")

    op_to_pre_edit_dates_path = f'{data_directory}/operation_to_pre_edit_dates_{mat_version}.pkl'
    if not (os.path.exists(op_to_pre_edit_dates_path)):
        print("saving op_to_pre_edit_dates")
        pruned_df = pd.read_feather(pruned_df_path)
        data_utils.save_operation_to_pre_edit_dates(pruned_df, op_to_pre_edit_dates_path)
        print("done saving op_to_pre_edit_dates")
    
    op_to_rep_coords_path = f'{data_directory}/operation_to_rep_coords_{mat_version}.pkl'
    if not (os.path.exists(op_to_rep_coords_path)):
        print("saving op_to_rep_coords")
        pruned_df = pd.read_feather(pruned_df_path)
        data_utils.save_operation_to_rep_coords(pruned_df, op_to_rep_coords_path)
        print("done saving op_to_rep_coords")

    op_to_pre_edit_roots_path = f'{data_directory}/operation_to_pre_edit_roots_{mat_version}.pkl'
    if not os.path.exists(op_to_pre_edit_roots_path):
        print("loading op_to_svs")
        op_to_svs = data_utils.load_pickle_dict(op_to_svs_path)
        print("loading op_to_pre_edit_dates")
        op_to_pre_edit_dates = data_utils.load_pickle_dict(op_to_pre_edit_dates_path)
        cpus = len(os.sched_getaffinity(0))
        print("num cpus", cpus)
        num_processes = cpus * 4
        print("num processes", num_processes)
        print("saving op_to_pre_edit_roots")
        data_utils.save_operation_to_pre_edit_roots(client, op_to_svs, op_to_pre_edit_dates, num_processes, op_to_pre_edit_roots_path)
        print("done saving op_to_pre_edit_roots")

    root_ids_txt_path = f'{data_directory}/pre_edit_roots_list_{mat_version}.txt'
    if not (os.path.exists(root_ids_txt_path)):
        print("saving roots txt file")
        op_to_pre_edit_roots = data_utils.load_pickle_dict(op_to_pre_edit_roots_path)
        op_to_pre_edit_roots_list = list(op_to_pre_edit_roots.values())
        data_utils.save_txt(root_ids_txt_path, op_to_pre_edit_roots_list)

    root_id_to_rep_coords_path = f'{data_directory}/root_id_to_rep_coords_{mat_version}.pkl'
    if not (os.path.exists(root_id_to_rep_coords_path)):
        print("saving root_id_to_rep_coords")
        op_to_rep_coords = data_utils.load_pickle_dict(op_to_rep_coords_path)
        op_to_pre_edit_roots = data_utils.load_pickle_dict(op_to_pre_edit_roots_path)
        data_utils.save_root_id_to_rep_coords(op_to_rep_coords, op_to_pre_edit_roots, root_id_to_rep_coords_path)


if __name__ == "__main__":
    main()