from auto_proof.code.pre import data_utils

import numpy as np
from tqdm import tqdm
import h5py
import graph_tool.all as gt
import sys
import multiprocessing

def __process_operation__(data):
    root = data
    root_path = f'/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/features_conf/{root}.hdf5'
    with h5py.File(root_path, 'r') as f:
        root_943s = f['root_943'][:]
    return root_943s, root

# Create the root 943 set in parallel
def parallel():
    roots = data_utils.load_txt('/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/post_label_roots_459972.txt')
    roots = roots[:500]
    manager = multiprocessing.Manager()
    # root_to_943_set = manager.dict()
    root_to_943_set = {}
    num_processes = 32
    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=len(roots)) as pbar:
        for root_943s, root in pool.imap_unordered(__process_operation__, roots):
            root_to_943_set[root] = set(root_943s)
            pbar.update()
    data_utils.save_pickle_dict('/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/dicts/root_to_943_set.pkl', root_to_943_set)
    print("key len", len(root_to_943_set.keys()))
    print("val len", len(root_to_943_set.values()))

# This currently doesn't work and I'm debugging
def create_graph(root_to_943_set):
    g = gt.Graph(directed=False)
    edges = []
    # all_vertices = []
    for root in root_to_943_set:
        curr_list = list(root_to_943_set[root])
        if len(curr_list) == 1:
            vertice = str(curr_list[0])
            edges.append((vertice, vertice))
            # all_vertices.append(vertice)
            # g.add_edge_list([(vertice, vertice)], hashed=True, hash_type="string")
            # print("root with one 943 root: root is ", root, "943 root is ", vertice)
        else:
            pairs = [(str(curr_list[i]), str(curr_list[i + 1])) for i in range(len(curr_list) - 1)]
            curr_vertices = [str(curr_list[i]) for i in range(len(curr_list))]
            # all_vertices.extend(curr_vertices)
            edges.extend(pairs)
            # g.add_edge_list(pairs, hashed=True, hash_type="string")
    
    # print("all unique vertices", len(all_vertices) == len(set(all_vertices)))
    # print("len of unique vertices", len(set(all_vertices)))
    vmap = g.add_edge_list(edges, hashed=True, hash_type="string")
    print("len of vertices from graph", len(g.get_vertices()))

    return g, vmap

if __name__ == "__main__":
    # parallel()

    root_to_943_set = data_utils.load_pickle_dict('/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/dicts/root_to_943_set.pkl')
    
    print("creating graph")
    g, vmap = create_graph(root_to_943_set)

    # Currently it only creates components for each separate root which would mean that none of the roots are shared
    comp, _ = gt.label_components(g)
    comp_arr = np.array(comp.a)
    # np.set_printoptions(threshold=sys.maxsize)
    # print(comp_arr)
    print("len of vertices from comp_arr", len(comp_arr))

    print("Creating root_943_to_cc dict")
    root_943_to_cc = {}
    for i in range(len(comp_arr)):
        root_943_to_cc[int(vmap[i])] = comp_arr[i]
    data_utils.save_pickle_dict('/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/dicts/root_943_to_cc.pkl', root_943_to_cc)

    print("Creating root_to_cc dict")
    root_to_cc = {}
    for root in root_to_943_set:
        root_943 = root_to_943_set[root].pop()
        root_to_cc[root] = root_943_to_cc[root_943]
    data_utils.save_pickle_dict('/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/dicts/root_to_cc.pkl', root_to_cc)

    print("Creating cc_to_root dict")
    cc_to_roots = {}
    for root in root_to_cc:
        cc = root_to_cc[root]
        if cc not in cc_to_roots:
            cc_to_roots[cc] = []
        cc_to_roots[cc].append(root)
    data_utils.save_pickle_dict('/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/dicts/cc_to_roots.pkl', cc_to_roots)

    print("Creating cc_roots")
    cc_roots = []
    for cc in cc_to_roots:
        cc_roots.append(cc_to_roots[cc])
    print(cc_roots[:3])
    data_utils.save_pickle_dict('/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/dicts/cc_roots.pkl', cc_roots)
