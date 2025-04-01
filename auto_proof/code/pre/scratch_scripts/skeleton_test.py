import caveclient as cc
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import graph_tool.all as gt
import h5py
import json
import sys
import time
import argparse
import glob

datastack_name = "minnie65_phase3_v1"
my_token = "64df9664ce8a28852ee99167c26a9e8d"
mat_version = 943
client = cc.CAVEclient(datastack_name=datastack_name, auth_token=my_token, version=mat_version)

print("cc version", cc.__version__)

sk = client.skeleton.get_skeleton(864691135359413848, skeleton_version=4, output_format="dict")
print(sk)