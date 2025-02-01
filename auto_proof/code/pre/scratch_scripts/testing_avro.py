from auto_proof.code.pre import data_utils

import numpy as np
import time
import h5py
import torch
from fastavro import writer, reader, parse_schema, load_schema

schema = load_schema('/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/avro/schema.avsc')

roots = data_utils.load_txt('/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/post_label_roots_459972.txt')
roots = roots[:100]

# root = roots[0]
roots = [roots[0]]

new_file_path = f'/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/avro/1_root.avro'
for root in roots:
    root_path = f'/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/features/{root}_1000.h5py'
    with h5py.File(root_path, 'r') as feature_file, open(new_file_path, 'wb') as new_file:        
        root_id = feature_file['root_id'][()]
        compartment = feature_file['compartment'][:].tolist()
        cutoff = feature_file['cutoff']
        edges = feature_file['edges'][:].tolist()
        num_initial_vertices = feature_file['num_initial_vertices'][()]
        num_vertices = feature_file['num_vertices'][()]
        pos_enc = feature_file['pos_enc'][:].tolist()
        radius = feature_file['radius'][:].tolist()
        rank_0 = feature_file['rank_0'][:].tolist()
        rank_1 = feature_file['rank_1'][:].tolist()
        rank_2 = feature_file['rank_2'][:].tolist()
        root_943 = feature_file['root_943'][:].tolist()
        vertices = feature_file['vertices'][:].tolist()
        label = feature_file['label'][:].tolist()
        confidence = feature_file['confidence'][:].tolist()
        confidence[label == 0] = 1

        record = {"root_id": root_id, "compartment": compartment, "cutoff": cutoff, "edges": edges, "num_initial_vertices": num_initial_vertices, "num_vertices": num_vertices, "pos_enc": pos_enc, "radius": radius, "rank_0": rank_0, "rank_1": rank_1, "rank_2": rank_2, "root_943": root_943, "vertices": vertices, "label": label, "confidence": confidence}
        writer(new_file, schema, record)

def read_one_root(root, file_path):
    with open('weather.avro', 'rb') as fo:
        for record in reader(fo):
            print(record)

# roots = [864691135778235581 (921), 864691135463516222 (1000), 864691135113539993 (1415), 864691135355969743 (1186), 864691134940631907 (1121)]
root = 864691135463516222

avg_time = 0
runs = 10
for i in range(runs):
    start = time.time()
    read_one_root(root, new_file_path)
    end = time.time()
    run_time = end - start
    avg_time += run_time
print("avg time for ", runs, " runs: ", avg_time / runs)


# Avro would not work for what I want