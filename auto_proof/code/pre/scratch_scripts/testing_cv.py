from cloudvolume import CloudVolume

root = 864691135463333789
seg_path = "graphene://middleauth+https://minnie.microns-daf.com/segmentation/table/minnie3_v1"
cv_seg = CloudVolume(seg_path, progress=False, use_https=True, parallel=True)
mesh = cv_seg.mesh.get(root, deduplicate_chunk_boundaries=False, remove_duplicate_vertices=False)
print(mesh)
mesh = mesh[root]
print(mesh)