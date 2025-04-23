import sys
import os
import numpy as np
from auto_proof.code.connectomics.reader import EmbeddingReader
from auto_proof.code.connectomics.sharding import md5_shard
import gcsfs

filesystem = gcsfs.GCSFileSystem(token='anon')
url = 'gs://iarpa_microns/minnie/minnie65/embeddings_m943/segclr_nm_coord_public_offset_csvzips/'

# num_shards = 10000
# bytewidth = 64

num_shards = 50000
bytewidth = 8

def sharder(segment_id: int, num_shards, bytewidth) -> int:
    return md5_shard(segment_id, num_shards=num_shards, bytewidth=bytewidth)

embedding_reader = EmbeddingReader(filesystem, url, sharder, num_shards, bytewidth)

# root = 864691135591041291
root = 864691134928303015

embs = embedding_reader[root]

emb_vals = []
for coord_key, emb_val in embs.items():
    emb_vals.append(emb_val)

emb_vals = np.array(emb_vals)

print("emb vals shape", emb_vals.shape)

print("coord key", coord_key)
print("mean emb", np.mean(emb_vals[:, 64:], axis=0))