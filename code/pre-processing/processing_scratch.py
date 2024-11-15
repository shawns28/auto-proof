from caveclient import CAVEclient
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from datetime import timedelta
import data_utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
if (torch.cuda.is_available()):
    print(torch.cuda.get_device_name(0))

df = pd.read_feather('../../data/240927/minnie_splitlog_240927.feather')
edits = np.load('../../data/240927/minnie_splitlog_240927.npy')
datastack_name = "minnie65_phase3_v1"
my_token = "64df9664ce8a28852ee99167c26a9e8d"
client = CAVEclient(datastack_name=datastack_name, auth_token=my_token)

op_to_pre_edit_roots = data_utils.load_pickle_dict('../../data/operation_to_pre_edit_roots_943.pkl')
reversed_dict = {value: key for key, value in op_to_pre_edit_roots.items()}
op = reversed_dict[864691135463333789]
print("op id", op)

def print_root_ids(df, operation_id, edits, materialization, client):
    # get source, sink svs for operation id
    indices = np.where(edits[:, 0] == operation_id)
    svs = edits[indices]
    # get first sink/source id for the operation id
    sink = svs[0][1]
    source = svs[0][2]
    first_svs = [sink, source]
    # get timestamps
    ts_edit = pd.to_datetime(df.loc[df['operation_id'] == operation_id, 'date'].values[0])
    one_ms = timedelta(milliseconds=1)
    ts_pre_edit = ts_edit - one_ms
    ts_mat = client.materialize.get_timestamp(version=materialization)
    print("pre-edit ts: ", ts_pre_edit)
    print("edit ts:     ", ts_edit)
    print("mat ts:      ", ts_mat)
    # get root ids
    counter = 0
    for sv in first_svs:
        root_id_pre_edit = client.chunkedgraph.get_root_id(sv, ts_pre_edit)
        root_id_edit = client.chunkedgraph.get_root_id(sv, ts_edit)
        root_id_mat =  client.chunkedgraph.get_root_id(sv, ts_mat)
        if counter == 0:
            print("sink sv:          ", sv)
        else:
            print("source sv:        ", sv)
        print("pre-edit root id: ", root_id_pre_edit)
        print("edit root id:     ", root_id_edit)
        print("mat root id:      ", root_id_mat)
        counter += 1

print_root_ids(df, op, edits, 943, client)
