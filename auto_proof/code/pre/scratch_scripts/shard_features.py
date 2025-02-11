from auto_proof.code.pre import data_utils

import json
import multiprocessing
import h5py
import os
from tqdm import tqdm
from time import sleep

CONFIG_PATH = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/base_config.json'
RETRIES = 5
'''
To copy over the features which are individual files currently and shard them into 64 files.
Also puts these new feature files in the new data folder inside of auto_proof
'''

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)
NUM_SHARDS = 10000
roots = data_utils.load_txt('/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/post_label_roots_459972.txt')
# roots = data_utils.load_txt('/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/missing_shard_roots_8.txt')

def hash_shard(root, num_shards):
    hash_value = hash(root)
    hash_value = abs(hash_value)
    shard_id = hash_value % num_shards
    return shard_id

def write_shard(root):
    # shard_id = hash_shard(root, NUM_SHARDS)
    # shard_path = f'/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/sharded_features_{NUM_SHARDS}_gzip_4/{shard_id}.hdf5'
    shard_path = f'/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/features_conf/{root}.hdf5'
    root_path = f'/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/features/{root}_1000.h5py'
    # Need this as well since it might be trying to write to the same shard as another process so we need to try again after a delay
    for i in range(RETRIES):
        try:
            with h5py.File(root_path, 'r') as feature_file, h5py.File(shard_path, 'a') as root_group:
            # with h5py.File(root_path, 'r') as feature_file, h5py.File(shard_path, 'a') as shard_file:
                # root_group = shard_file.create_group(str(root))
                for name, obj in feature_file.items():
                    if name != 'confidence' and name != 'label':
                        if isinstance(obj, h5py.Dataset):
                            if obj.size == 1:
                                data = obj[()]
                                root_group.create_dataset(name, data=data)
                            else:
                                data = obj[:]
                                root_group.create_dataset(name, data=data)
                                # root_group.create_dataset(name, data=data, compression="gzip", compression_opts=4)
                            # root_group.create_dataset(name, data=data, chunks=True)
                label = feature_file['label'][:]
                root_group.create_dataset('label', data=label)
                confidence = feature_file['confidence'][:]
                confidence[label == 0] = 1
                root_group.create_dataset('confidence', data=confidence)

            with open(f'/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/successful_sharded_roots_features_conf/{root}', 'w') as f:
                pass
            break
        except Exception as e:
            if str(e) == "Unable to create group (name already exists)":
                break
            if i == RETRIES - 1:
                print(type(e).__name__)
                print("Failed to shard root: ", root, " error: ", e)
            sleep(5)
            continue
    
    # Testing the read
    # with h5py.File(shard_path, 'r') as read_file:
    #     root_group = read_file[str(root)]
    #     vertices = root_group['vertices'][:]
    #     root_id = root_group['root_id'][()]
    #     print(vertices)
    #     print(root_id)
    
# Root id: 864691135778235581, 921 vertices
# write_shard(864691135778235581)

# num_processes = os.cpu_count()
# num_processes = 1
num_processes = 32
print("The node has", os.cpu_count(), "and we are using", num_processes)
with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
    for _ in pool.imap_unordered(write_shard, roots):
        pbar.update()