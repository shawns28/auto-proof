from auto_proof.code.pre import data_utils

import numpy as np
import time
import h5py
import torch

roots = data_utils.load_txt('/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/proofread_943.txt')
# roots = roots[:1]

# root = roots[150]
root = 864691135294197260
print(root)

# roots = [roots[0]]


new_file_path = f'/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/proofread_features/{root}.hdf5'

# for root in roots:
#     root_path = f'/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/features/{root}_1000.h5py'
#     with h5py.File(root_path, 'r') as feature_file, h5py.File(new_file_path, 'a') as new_file:
#         root_group = new_file.create_group(str(root))
#         for name, obj in feature_file.items():
#             if name != 'confidence' and name != 'label':
#                 if isinstance(obj, h5py.Dataset):
#                     if obj.size == 1:
#                         data = obj[()]
#                         root_group.create_dataset(name, data=data)
#                     else:
#                         data = obj[:]
#                         root_group.create_dataset(name, data=data, compression="gzip", compression_opts=4)
#                     # root_group.create_dataset(name, data=data, chunks=True)
#         label = feature_file['label'][:]
#         root_group.create_dataset('label', data=label, compression="gzip", compression_opts=4)
#         confidence = feature_file['confidence'][:]
#         confidence[label == 0] = 1
#         root_group.create_dataset('confidence', data=confidence, compression="gzip", compression_opts=4)

# Took out some stuff, just want to see basic read time difference on numpy
def read_one_root(root, file_path):
    print(file_path)
    with h5py.File(file_path, 'r') as read_new_file:
        try: 
            # f = read_new_file[str(root)]
            f = read_new_file
            vertices = f['vertices'][:]
            print(vertices)
            compartment = f['compartment'][:]
            print(compartment)
            radius = f['radius'][:]
            print(radius)
            # pos_enc = f['pos_enc'][:]
            # labels = f['label'][:]
            # confidence = f['confidence'][:]
            num_vertices = f['num_vertices'][()]

            # rank = f['rank_1'][:]

            size = len(vertices)
            # print("size", size)

            edges = f['edges'][:]
            print(edges)

        except Exception as e:
            print("root: ", root, "error: ", e)

# roots = [864691135778235581 (921), 864691135463516222 (1000), 864691135113539993 (1415), 864691135355969743 (1186), 864691134940631907 (1121)]
# test_roots = [864691135778235581, 864691135463516222, 864691135113539993, 864691135355969743, 864691134940631907]
# test_roots = [864691135778235581]
test_roots = [root]

avg_time = 0
runs = 10
for i in range(runs):
    for root in test_roots:
        start = time.time()
        read_one_root(root, new_file_path)
        end = time.time()
        run_time = end - start
        avg_time += run_time
print("avg time for ", runs, " runs: ", avg_time / (runs * len(test_roots)))


''' Pytorch version
with h5py.File(new_file_path, 'r') as read_new_file:
    try: 
        f = read_new_file[str(root)]
        vertices = torch.from_numpy(f['vertices'][:])
        compartment = torch.from_numpy(f['compartment'][:]).unsqueeze(1)
        radius = torch.from_numpy(f['radius'][:]).unsqueeze(1)
        pos_enc = torch.from_numpy(f['pos_enc'][:])
        labels = torch.from_numpy(f['label'][:])
        confidence = torch.from_numpy(f['confidence'][:])

        labels = labels.unsqueeze(1).int()
        confidence = confidence.unsqueeze(1).int()

        # If doing the default features, need to set confidence to 1 where labels are 0
        # confidence[label == 0] = 1

        # Not adding rank as a feature
        rank_num = f'rank_{seed_index}'
        rank = torch.from_numpy(f[rank_num][:])

        size = len(vertices)

        # Currently concatenating the pos enc to node features
        # Should remove the ones if not going to predict the padded ones
        # SHOULD REMOVE THE ONES NOW SINCE I USE -1 later to indicate that the label is padded anyway
        input = torch.cat((vertices, compartment, radius, pos_enc, torch.ones(size, 1)), dim=1)

        edges = f['edges'][:]

    except Exception as e:
        print("root: ", root, "error: ", e)
'''

'''
hash took 0.00000620. sec so this isn't the issue

root: 864691135778235581 (first root)
avg read time with only its group in the file: 0.009 sec
file size was 316 kb for the hdf5 file with just this root

hdf5 file with 100 root groups in it is now 34.2 mb
The first root took 0.016 sec on average of 5 attempts and I also got 0.022
So it was roughly 1.5 times slower
Second root: 864691135463516222 took 0.02 sec which is 2x as slow
Third root: 864691135113539993 took 0.0188
4th root: 864691135355969743 0.0175
5Th root and last in file: 864691134940631907 0.01 which is same as original


first root with no group by itself in its own file: 0.009
first root with no group in a file with 100 is 0.029 which is like 3x as slow
2nd root: 0.02759542465209961 which is also 3x as slow
This makes sense to me since the time it takes to find the group will be slower

first_100 group
1 0.013
100 0.014
5 avg 0.016

first 100 no group
bad

autochunk
first root 0.009
first root with 100 0.014
5 roots avg in the 100 0.02

gzip 1
first root 0.01
first root with 100 0.01
5 roots avg in the 100 0.014

gzip 4
firs root 0.009
first root with 100 0.01
5 roots avg in the 100 0.015
'''