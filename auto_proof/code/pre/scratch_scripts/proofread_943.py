from auto_proof.code.pre import data_utils

from caveclient import CAVEclient
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
import json
import sys
import time

def main():
    df = pd.read_csv('../../data/proofread_943.csv')
    print(df.head())
    filtered_df = df[df['status_axon'] != 'non']
    print(filtered_df.head())
    root_ids = filtered_df['root_id']
    print(root_ids)
    root_ids_array = np.array(root_ids)
    print(root_ids_array)
    data_utils.save_txt('../../data/proofread_943.txt', root_ids_array)

if __name__ == "__main__":
    main()