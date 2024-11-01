import data_utils
from caveclient import CAVEclient
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
import json
import sys
import time
import glob
import dataset
from cloudvolume import CloudVolume
import multiprocessing
import time
import matplotlib.pyplot as plt

def __save_root943__(root_path):
    # seg_path = "graphene://middleauth+https://minnie.microns-daf.com/segmentation/table/minnie3_v1"
    seg_path = "precomputed://gs://iarpa_microns/minnie/minnie65/seg_m943/"
    cv_seg = CloudVolume(seg_path, use_https=True)
    # print("root path", root_path)
    root = root_path[-28:-10]
    resolution = np.array([[8, 8, 40]])
    with h5py.File(root_path, 'r+') as f:
        vertices = f['vertices'][:]
        # print("og vertices", vertices)
        vertices = vertices / resolution
        input_vertices = [(vertices[i][0].item(), vertices[i][1].item(), vertices[i][2].item()) for i in range(len(vertices))]
        for i in range(2):   
            try:
                root_943_dict = cv_seg.scattered_points(input_vertices)
                break
            except Exception as e:
                print("Failed to get scattered points for root", root, "error", e)
                if i == 0:
                    continue
                else:
                    with open(f'../../data/error_root_943/{root}', 'w') as f:
                        return
        root_943_arr = np.array([root_943_dict[input_vertices[i]] for i in range(len(vertices))])
        f.create_dataset('root_943', data=root_943_arr)
        with open(f'../../data/successful_root_943/{root}', 'w') as f:
            pass
    
def svs_parallel(args):
    data_directory = '../../data'
    features_directory = f'{data_directory}/features'
    # features_directory = f'{data_directory}/features_copy/features'
    root_paths = glob.glob(f'{features_directory}/*')
    # very bad because this is based off of the path which could always change
    
    chunk_num = int(args[0])
    num_chunks = 24
    print("chunk number is:", chunk_num)
    print("num_chunks is:", num_chunks)
    chunk_size = len(root_paths) // num_chunks
    start_index = (chunk_num - 1) * chunk_size
    end_index = start_index + chunk_size + 1
    if chunk_num == num_chunks:
        root_paths = root_paths[start_index:]
    else:
        root_paths = root_paths[start_index:end_index]

    num_processes=64

    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(root_paths)) as pbar:
        for _ in pool.imap_unordered(__save_root943__, root_paths):
            pbar.update()
   

def svs_loader():
    seg_path = "graphene://middleauth+https://minnie.microns-daf.com/segmentation/table/minnie3_v1"
    cv_seg = CloudVolume(seg_path, use_https=True, fill_missing=True)
    print(cv_seg.resolution)
    with open('../configs/base_config.json', 'r') as f:
        config = json.load(f)
    data = dataset.VerticesConvertedDataset(config)
    loader = dataset.build_dataloader(data)
    resolution = np.array([[8, 8, 40]])
    with tqdm(total=data.__len__()) as pbar:
        for batch in loader:
            root, vertices = batch
            root = root[0]
            # print("root", root)
            vertices = vertices[0]
            print(len(vertices))
            vertices = vertices / resolution
            # print("vertices", vertices)
            svs = np.empty(len(vertices), dtype=int)
            for i, vertice in enumerate(vertices):
                vol = cv_seg[vertice[0].item(), vertice[1].item(), vertice[2].item()]
                svs[i] = vol[0][0][0][0]
            # print(svs)
            pbar.update()

def svs_loader_parallel():
    with open('../configs/base_config.json', 'r') as f:
        config = json.load(f)
    data = dataset.VerticesParallelDataset(config)
    loader = dataset.build_dataloader(data)
    with tqdm(total=data.__len__()) as pbar:
        for batch in loader:
            data = batch
            pbar.update()

def one_iteration():
    start_time = time.time()
    # 864691135939883265
    root_path = "../../data/features/864691135939883265_1000.h5py"
    print("root path", root_path)
    
    # seg_path = "graphene://middleauth+https://minnie.microns-daf.com/segmentation/table/minnie3_v1"
    seg_path = "precomputed://gs://iarpa_microns/minnie/minnie65/seg_m943/"
    print("seg path", seg_path)
    
    # Using secrets below directly didn't work
    # json_token = {
    #     "token": "64df9664ce8a28852ee99167c26a9e8d"
    # }
    # json_string = json.dumps(json_token)
    # cv_seg = CloudVolume(seg_path, progress=True, secrets=json_string)
    
    # cv_seg = CloudVolume(seg_path, use_https=True)
    # Just hangs because it needs credentials
    # cv_seg = CloudVolume(seg_path, progress=False, parallel=True)
    # cv_seg = CloudVolume(seg_path, progress=False, use_https=True)
    cv_seg = CloudVolume(seg_path, progress=False, use_https=True, parallel=True)
    
    # Both are 8, 8, 40
    print("seg reso", cv_seg.resolution)
    resolution = np.array([[8, 8, 40]])
    
    with h5py.File(root_path, 'r') as f:
        vertices = f['vertices'][:]
        print(len(vertices))
        # print("og vertices", vertices)
        vertices = vertices / resolution
        # print("reso vertices", vertices)
        svs = np.empty(len(vertices), dtype=int)
        svs_time = time.time()
        # sv_time_arr = np.empty(len(vertices))
        
        input_vertices = [(vertices[i][0].item(), vertices[i][1].item(), vertices[i][2].item()) for i in range(len(vertices))]
        root_943s = cv_seg.scattered_points(input_vertices)
        print("first vertice root 943", root_943s[input_vertices[0]])
        svs = [root_943s[input_vertices[i]] for i in range(len(vertices))]
        # for i, vertice in enumerate(vertices):
        #     sv_start = time.time()
        #     vol = cv_seg.download_point((vertice[0].item(), vertice[1].item(), vertice[2].item()), size=1)
        #     svs[i] = vol.squeeze()
        #     sv_end = time.time()
        #     sv_time_arr[i] = sv_end - sv_start

        end_time = time.time()
        print("total time", end_time - start_time)
        print("svs time", end_time - svs_time)
        print("svs", svs)

        # x = np.arange(len(vertices))
        # plt.plot(x, sv_time_arr)
        # plt.xlabel('Index')
        # plt.ylabel('Time')
        # plt.title('How long each vertice takes')
        # plt.savefig('../../data/figures/cv_vertice_times_943seg_root864691135939883265_scatterpoints')
        # plt.show()

def svs_sequential():
    seg_path = "graphene://middleauth+https://minnie.microns-daf.com/segmentation/table/minnie3_v1"
    cv_seg = CloudVolume(seg_path, use_https=True, fill_missing=True)
    print(cv_seg.resolution)
    # data = dataset.VerticesDataset(config)
    # loader = dataset.build_dataloader(data)
    data_directory = '../../data'
    #features_directory = f'{data_directory}/features'
    features_directory = f'{data_directory}/features_copy/features'
    root_paths = glob.glob(f'{features_directory}/*')
    # very bad because this is based off of the path which could always change
    print(root_paths[0])
    root_paths = root_paths[:5]
    resolution = np.array([[8, 8, 40]])
    with tqdm(total=len(root_paths)) as pbar:
        for root_path in root_paths:
            print(root_path)
            with h5py.File(root_path, 'r+') as f:
                vertices = f['vertices'][:]
                print(len(vertices))
                # print("og vertices", vertices)
                vertices = vertices / resolution
                # print("reso vertices", vertices)
                svs = np.empty(len(vertices), dtype=int)
                print("about to do svs")
                for i, vertice in enumerate(vertices):
                    vol = cv_seg[vertice[0].item(), vertice[1].item(), vertice[2].item()]
                    svs[i] = vol[0][0][0][0]
                pbar.update()

def test_h5_add():
    hf = h5py.File(f'{data_directory}/num_vertices.hdf5', 'r+')
    root = '111111111111111111'
    num_vertices = [[11]]
    hf.create_dataset(root, data=num_vertices)
    hf.close()
    print("done adding dataset")

    # root = '864691135815797071'
    rf = h5py.File(f'{data_directory}/num_vertices.hdf5', 'r')
    print(rf[root][()])
    num_vertices = rf[root][()]
    print(num_vertices[0][0].dtype)

def get_roots_467853():
    files = glob.glob(f'{features_directory}/*')
    print(files[0])
    file = files[0][20:38]
    print(file)
    roots = [files[i][20:38] for i in range(len(files))]
    data_utils.save_txt(f'{data_directory}/roots_467853.txt', roots)


if __name__ == "__main__":
    # test_h5_add()
    # svs_sequential()
    # svs_loader()
    svs_parallel(sys.argv[1:])
    #svs_loader_parallel() 
    # one_iteration()
