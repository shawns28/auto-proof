## Configuration Structure and Parameters

The configuration is organized into several top-level sections, each representing a distinct part of the data processing pipeline.

### `data_dir`
* **Type:** `string`
* **Description:** The absolute path to the base directory where all project data (raw edits, proofread data, generated skeletons, features, labels, segmentation, and segclr data) is stored. All other data-related paths are typically relative to this directory.

### `multiprocessing`
* **Description:** Settings for parallel processing, primarily used in data generation and preprocessing steps.
    * `num_chunks`
        * **Type:** `integer`
        * **Description:** The number of data chunks to divide the data into. 
    * `num_processes`
        * **Type:** `integer`
        * **Description:** The maximum number of parallel processes to use for multiprocessing tasks.

### `raw_edits`
* **Description:** Configuration for processing and managing raw edits.
    * `intermediate_version`
        * **Type:** `integer`
        * **Description:** An intermediate materialization version to help with SegCLR versioning. Shouldn't be the start or end materialization version (from client_config) but something inbetween. Ex: 943 if start is 343 and end is 1300.
    * `raw_df`
        * **Type:** `string`
        * **Description:** Absolute path to the feather file containing the raw split log dataframe.
    * `raw_edit`
        * **Type:** `string`
        * **Description:** Absolute path to the NumPy array file containing supervoxel raw edits.
    * `root_to_rep`
        * **Type:** `string`
        * **Description:** Filename within `data_dir` for a pickle file mapping root IDs to their representative edit coordinates.
    * `post_raw_edit_roots`
        * **Type:** `string`
        * **Description:** Filename within `roots_dir` (created based on the start and end mat version in client_config) for a text file listing roots that have been processed for raw edits.

### `proofread`
* **Description:** Configuration for managing fully proofread skeleton information.
    * `proofread_dir`
        * **Type:** `string`
        * **Description:** Directory within `data_dir` where proofread root lists and CSVs are stored.
    * `post_proofread_roots`
        * **Type:** `string`
        * **Description:** Filename within `proofread_dir` for a text file listing roots that have been identified as proofread.
    * `csv_943`
        * **Type:** `string`
        * **Description:** Absolute path to the CSV file containing proofread root IDs for materialization version 943.
    * `csv_1300`
        * **Type:** `string`
        * **Description:** Absolute path to the CSV file containing proofread root IDs for materialization version 1300.
    * `mat_versions`
        * **Type:** `list of integers`
        * **Description:** A list of materialization versions for which fuly proofread roots are used.
    * `copy_count`
        * **Type:** `integer`
        * **Description:** Number of copies/duplicates to create for proofread examples during dataset balancing or augmentation.

### `generate_skels`
* **Description:** Settings for the process of generating or retrieving skeleton data.
    * `post_generate_roots`
        * **Type:** `string`
        * **Description:** Filename within `roots_dir` for a text file listing roots for which skeletons have been successfully generated.
    * `generate_chunk_size`
        * **Type:** `integer`
        * **Description:** The number of skeletons to generate in a single chunk or batch during the generation process.
    * `exists_chunk_size`
        * **Type:** `integer`
        * **Description:** The chunk size used when checking for the existence of already generated skeletons.

### `features`
* **Description:** Configuration for extracting and processing graph features from skeletons.
    * `features_dir`
        * **Type:** `string`
        * **Description:** Directory within `data_dir` where processed feature HDF5 files are stored.
    * `post_feature_roots`
        * **Type:** `string`
        * **Description:** Filename within `roots_dir` for a text file listing roots for which features have been successfully saved.
    * `cutoff`
        * **Type:** `integer`
        * **Description:** Represents the number of vertices that are kept from each root based on the k (`cutoff`) lowest rank nodes. If the root has less than `cutoff` nodes then nothing is pruned.
    * `skeleton_version`
        * **Type:** `integer`
        * **Description:** The specific version of the skeletonization service.
    * `pos_enc_dim`
        * **Type:** `integer`
        * **Description:** The dimensionality of the positional encoding feature.
    * `box_cutoff`
        * **Type:** `integer`
        * **Description:** The k (box_cutoff) lowest rank nodes with a guarantee that that the representitive edit if it exists will be within this bounding box/core.

### `labels`
* **Description:** Configuration for generating ground truth labels for proofreading errors.
    * `latest_mat_version`
        * **Type:** `integer`
        * **Description:** The latest materialization version used for defining ground truth labels.
    * `roots_at_latest_dir`
        * **Type:** `string`
        * **Description:** Directory prefix within `data_dir` for root lists specific to the latest materialization version (e.g., `roots_at_1300/`).
    * `labels_at_latest_dir`
        * **Type:** `string`
        * **Description:** Directory prefix within `data_dir` for label files specific to the latest materialization version (e.g., `labels_at_1300/`).
    * `labels_type`
        * **Type:** `string`
        * **Description:** Defines the type of labeling strategy. `labels_type` can be one of 3 options:
            * **ignore_inbetween** Ignores errors where each side of the error component belongs to the same component (baseline).
            * **ignore_inbetween_and_edge** Ignores above and errors on the edge of a root (where the error component only has one neighbor).
            * **ignore_nothing** Deprecated but represented the original labels.
    * `ignore_edge_ccs`
        * **Type:** `boolean`
        * **Description:** If `true`, ignore connected components that are on the edge of a root (where the error component only has one neighbor).
    * `post_label_roots`
        * **Type:** `string`
        * **Description:** Filename within `roots_dir` for a text file listing roots for which labels have been successfully saved.

### `segmentation`
* **Description:** URLs and resolutions for accessing precomputed segmentation volumes.
    * `precomputed_343`
        * **Type:** `string`
        * **Description:** Neuroglancer precomputed URL for segmentation data at materialization version 343.
    * `precomputed_943`
        * **Type:** `string`
        * **Description:** Neuroglancer precomputed URL for segmentation data at materialization version 943.
    * `precomputed_1300`
        * **Type:** `string`
        * **Description:** Neuroglancer precomputed URL for segmentation data at materialization version 1300.
    * `resolution`
        * **Type:** `list of integers`
        * **Description:** The spatial resolution of the segmentation data in X, Y, Z (e.g., in nanometers per voxel).

### `segclr`
* **Description:** Configuration for segmentation color (embedding) features.
    * `url_343`
        * **Type:** `string`
        * **Description:** Google Cloud Storage URL for SegCLR embeddings at materialization version 343.
    * `bytewidth_343`
        * **Type:** `integer`
        * **Description:** Byte width of the SegCLR embeddings for version 343.
    * `num_shards_343`
        * **Type:** `integer`
        * **Description:** Number of shards for the SegCLR embeddings at version 343.
    * `url_943`
        * **Type:** `string`
        * **Description:** Google Cloud Storage URL for SegCLR embeddings at materialization version 943.
    * `bytewidth_943`
        * **Type:** `integer`
        * **Description:** Byte width of the SegCLR embeddings for version 943.
    * `num_shards_943`
        * **Type:** `integer`
        * **Description:** Number of shards for the SegCLR embeddings at version 943.
    * `emb_dim`
        * **Type:** `integer`
        * **Description:** The first k dimensions to use from the SegCLR embedding vectors.
    * `roots_at_segclr_dir`
        * **Type:** `string`
        * **Description:** Directory prefix within `data_dir` for root lists related to segclr processing.
    * `segclr_dir`
        * **Type:** `string`
        * **Description:** Directory within `data_dir` where processed segclr HDF5 files are stored.
    * `post_segclr_roots`
        * **Type:** `string`
        * **Description:** Filename within `roots_dir` for a text file listing roots for which segclr features have been successfully saved.
    * `mat_versions`
        * **Type:** `list of integers`
        * **Description:** A list of materialization versions for which SegCLR embeddings exist and are used.
    * `small_radius`
        * **Type:** `integer`
        * **Description:** A smaller radius value used for SegCLR node mapping onto L2 nodes.
    * `large_radius`
        * **Type:** `integer`
        * **Description:** A larger radius value used to determine the max radius we can use for assigning SegCLR nodes to L2 nodes.
    * `visualize_radius`
        * **Type:** `boolean`
        * **Description:** If `true`, enable visualization features related to segclr radius for PyVista visualization.

### `split`
* **Description:** Configuration for splitting the dataset into training, validation, and test sets.
    * `split`
        * **Type:** `list of floats`
        * **Description:** A list representing the proportions for the train, validation, and test splits, respectively. Values should sum to 1.0.
        * **Example:** `[0.8, 0.1, 0.1]` for 80% train, 10% validation, 10% test.
    * `split_dir`
        * **Type:** `string`
        * **Description:** Directory prefix within the `roots_dir` where generated split files (e.g., `train_roots.txt`) will be stored.