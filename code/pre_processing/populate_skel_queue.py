import data_utils
import caveclient as cc
import os
import pandas as pd
import numpy as np
import pickle
import json
from tqdm import tqdm
import h5py
from timeit import default_timer
from time import sleep
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import sys

def main(args):
    chunk_index = int(args[0]) - 1
    # num_chunks = int(args[1])
    num_chunks = 8
    print("chunk index is:", chunk_index)
    print("num_chunks is:", num_chunks)
    with open("../../data/Shawn_Stanley__pre_edit_roots_list_943__remaining.txt") as f:
        root_ids_txt = f.read()
    root_ids = [int(v) for v in root_ids_txt.strip().split('\n')]
    chunk_size = len(root_ids) // num_chunks
    start_index = chunk_index * chunk_size
    end_index = start_index + chunk_size + 1
    if chunk_index + 1 == num_chunks:
        root_ids = root_ids[start_index:]
    else:
        root_ids = root_ids[start_index:end_index]

    root_ids_len = len(root_ids)
    server = "minnie"
    skvn = 2
    # sample_root_id = root_ids[1]  # Arbitrarily, use the first root id for this test

    # url = f"https://{server}.microns-daf.com/skeletoncache/api/v1/minnie65_phase3_v1/precomputed_via_msg/skeleton/{skvn}/{sample_root_id}"
    # print(url)

    auth_client = cc.auth.AuthClient(token="64df9664ce8a28852ee99167c26a9e8d")
    auth_header = auth_client.request_header
    print("auth_header:", auth_header)

    t0 = default_timer()
    print(f"Start time: {t0}")
    t_prev = t0
    et_accum = 0
    num_skeletons_requested = 0
    MAX_ATTEMPTS = 10
    for i, root_id in enumerate(root_ids):
        # if i % 10 == 0:
        #     print(f"[{i%100}]", end='')  # square brackets indicate the count of root ids being considered

        url = f"https://{server}.microns-daf.com/skeletoncache/api/v1/minnie65_phase3_v1/precomputed_via_msg/skeleton/{skvn}/{root_id}"

        # Make a limited number of attempts to process the current root id
        num_attempts = 0
        while True:
            num_attempts += 1
            if num_attempts > MAX_ATTEMPTS:
                "Too many failed attempts for one root id. Aborting..."
                break
            
            try:
                # THIS IS IT! This is where we queue up a new root id for skeletonization.
                r = requests.get(url, verify=False, headers=auth_header, timeout=10)
            except requests.exceptions.Timeout:
                print("\nrequests.get() timeout")
                continue
            except Exception as e:
                print("\nException in requests.get(): ", e)
                continue
        
            if r.status_code == 200:  # Success
                break
                
            print("\nERROR:", r.status_code, r.headers['content-type'], r.encoding)

            pause_s = 1
            print(f"Sleeping for {pause_s} seconds and trying again...")
            sleep(pause_s)

        if num_attempts > MAX_ATTEMPTS:
            print("Max attempts exceeded for one root id! Aborting...")
            break

        # if num_skeletons_requested % 10 == 0:
        #     print(f"<{num_skeletons_requested%100}>", end='')  # angle brackets indicate the count of new messages dispatched
        # else:
        #     print(".", end='')
        
        num_skeletons_requested += 1
        if num_skeletons_requested % 100 == 0:
            t = default_timer()
            et_prev = t - t_prev
            et0 = t - t0
            t_prev = t
            et_accum += et_prev
            secs_per_request_prev = et_prev / 100
            requests_per_sec_prev = 100 / et_prev
            secs_per_request_overall = et0 / num_skeletons_requested
            requests_per_sec_overall = num_skeletons_requested / et0
            mins_per_100requests_overall = (secs_per_request_overall * 100) / 60
            mins_per_1000requests_overall = (secs_per_request_overall * 1000) / 60
            print(f"Processed {i} out of {root_ids_len}")
            print(f"\nPrev-100: Secs: {et_prev:3.1f}    Secs_accum: {et_accum:3.1f}    Requests / sec: {requests_per_sec_prev:3.1f}")
            print(f"Overall:                Secs:       {et0:3.1f}    Requests / sec: {requests_per_sec_overall:3.1f}    Mins / 100 requests: {mins_per_100requests_overall:8.1f}    Mins / 1000 requests: {mins_per_1000requests_overall:8.1f}")
            print(f"Current time: {t}\n")

    t = default_timer()
    et0 = t - t0
    secs_per_request_overall = et0 / num_skeletons_requested
    requests_per_sec_overall = num_skeletons_requested / et0
    mins_per_1000requests_overall = (secs_per_request_overall * 1000) / 60
    print("\n\nFINAL RESULTS")
    print(f"End time: {t}")
    print(f"{num_skeletons_requested} requests in {et0:.1f}s ({et0/60:.1f}min)")
    print(f"Requests / sec: {requests_per_sec_overall:3.1f}    Mins / 1000 requests: {mins_per_1000requests_overall:8.1f}")

    print("Done\n")

if __name__ == "__main__":
    main(sys.argv[1:])
