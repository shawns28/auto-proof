import sys
import os
import numpy as np
from auto_proof.code.connectomics.reader import EmbeddingReader
from auto_proof.code.connectomics.sharding import md5_shard
import gcsfs

filesystem = gcsfs.GCSFileSystem(token='anon')
url = 'gs://iarpa_microns/minnie/minnie65/embeddings_m1300/segclr_nm_coord_public_offset_csvzips/'

num_shards = 50_000
bytewidth = 8

def sharder(segment_id: int) -> int:
    return md5_shard(segment_id, num_shards=num_shards, bytewidth=bytewidth)

embedding_reader = EmbeddingReader(filesystem, url, sharder)

root_943_id = 864691135591041291
embs = embedding_reader[root_943_id]

emb_vals = []
for coord_key, emb_val in embs.items():
    emb_vals.append(emb_val)

emb_vals = np.array(emb_vals)

print("emb vals shape", emb_vals.shape)

print("coord key", coord_key)