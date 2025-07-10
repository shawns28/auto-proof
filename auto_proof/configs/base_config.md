## Configuration Structure and Parameters

The configuration is organized into several top-level sections, each representing a distinct part of the model's operation.

### `data`
* **Description:** General data-related paths and parameters relevant to the dataset loading.
    * `data_dir`
        * **Type:** `string`
        * **Description:** The absolute path to the base directory where all project data is stored. All other data-related paths within this section are typically relative to this directory.
    * `features_dir`
        * **Type:** `string`
        * **Description:** Directory within `data_dir` where processed graph feature HDF5 files are located.
    * `labels_dir`
        * **Type:** `string`
        * **Description:** Directory within `data_dir` where ground truth label HDF5 files are located. The name often reflects the materialization version and labeling strategy (e.g., `labels_at_1300_ignore_inbetween/`).
    * `segclr_dir`
        * **Type:** `string`
        * **Description:** Directory within `data_dir` where processed SegCLR embedding HDF5 files are stored.
    * `proofread_dir`
        * **Type:** `string`
        * **Description:** Directory within `data_dir` where proofread root lists and related CSVs are located.
    * `split_dir`
        * **Type:** `string`
        * **Description:** Directory within `roots_dir` (which is constructed from `data_dir` and `mat_version_start`/`end`) where dataset split files (e.g., `train_roots.txt`) are stored. The name often includes a unique identifier like `split_598963/`.
    * `all_roots`
        * **Type:** `string`
        * **Description:** Filename within `split_dir` containing a list of all roots in the dataset.
    * `train_roots`
        * **Type:** `string`
        * **Description:** Filename within `split_dir` containing the list of roots designated for the training set.
    * `val_roots`
        * **Type:** `string`
        * **Description:** Filename within `split_dir` containing the list of roots designated for the validation set.
    * `test_roots`
        * **Type:** `string`
        * **Description:** Filename within `split_dir` containing the list of roots designated for the test set.
    * `proofread_roots`
        * **Type:** `string`
        * **Description:** Filename within `proofread_dir` containing a consolidated list of proofread roots from specified materialization versions (e.g., `943_1300.txt`).
    * `obj_det_val_roots`
        * **Type:** `string`
        * **Description:** Filename within `split_dir` containing a specific subset of validation roots used for object detection metrics, potentially filtered for specific error conditions.
    * `mat_version_start`
        * **Type:** `integer`
        * **Description:** The starting materialization version used to define the range of root IDs or data.
    * `mat_version_end`
        * **Type:** `integer`
        * **Description:** The ending materialization version used to define the range of root IDs or data.
    * `box_cutoff`
        * **Type:** `integer`
        * **Description:** The k (box_cutoff) lowest rank nodes with a guarantee that the representative edit, if it exists, will be within this bounding box/core. Used for prediction metrics.

### `loader`
* **Description:** Parameters specifically for configuring the PyTorch `DataLoader`, which manages how data samples are loaded and batched.
    * `num_workers`
        * **Type:** `integer`
        * **Description:** The number of subprocesses to use for data loading. Higher values can speed up loading but use more memory.
    * `prefetch_factor`
        * **Type:** `integer`
        * **Description:** Number of samples loaded in advance by each worker. If 0, prefetching is disabled.
    * `batch_size`
        * **Type:** `integer`
        * **Description:** The number of samples per batch to load.
    * `fov`
        * **Type:** `integer`
        * **Description:** Field of View; the maximum number of vertices (nodes) to consider for each graph sample. Graphs larger than this will be truncated/pruned. Graphs smaller in size are given padding nodes to reach the `fov` size.
    * `feat_dim`
        * **Type:** `integer`
        * **Description:** The expected total dimensionality of the input features per vertex after concatenation (e.g., coordinates + radius + positional encoding + segclr + has_emb).
    * `use_segclr`
        * **Type:** `boolean`
        * **Description:** If `true`, include SegCLR embeddings as part of the input features.
    * `relative_vertices`
        * **Type:** `boolean`
        * **Description:** If `true`, vertices are normalized to be relative to their mean coordinates within each sample.
    * `zscore_radius`
        * **Type:** `boolean`
        * **Description:** If `true`, apply z-score normalization (mean 0, std 1) to the radius feature.
    * `zscore_segclr`
        * **Type:** `boolean`
        * **Description:** If `true`, apply z-score normalization to the SegCLR features.
    * `l2_norm_segclr`
        * **Type:** `boolean`
        * **Description:** If `true`, apply L2 normalization to the SegCLR features.
    * `zscore_pe`
        * **Type:** `boolean`
        * **Description:** If `true`, apply z-score normalization to the positional encoding (PE) features.

### `model`
* **Description:** Architectural parameters for the AutoProof graph neural network model.
    * `num_classes`
        * **Type:** `integer`
        * **Description:** The number of output classes for the model's prediction (e.g., 1 for binary classification like error/no error).
    * `dim`
        * **Type:** `integer`
        * **Description:** The hidden dimension (embedding size) used throughout the model's layers.
    * `depth`
        * **Type:** `integer`
        * **Description:** The number of transformer layers in the model.
    * `n_head`
        * **Type:** `integer`
        * **Description:** The number of attention heads to use in multi-head attention mechanisms.

### `trainer`
* **Description:** Configuration for the training process, including checkpoints, visualization, and loss weighting.
    * `ckpt_dir`
        * **Type:** `string`
        * **Description:** Directory where model checkpoints will be saved.
    * `save_ckpt_every`
        * **Type:** `integer`
        * **Description:** Save a model checkpoint every N epochs.
    * `visualize_rand_num`
        * **Type:** `integer`
        * **Description:** A seed or random number to use for selecting samples for visualization during training.
    * `visualize_cutoff`
        * **Type:** `integer`
        * **Description:** A root is only considered for visualization if the number of L2 nodes is less than the `visualize_cutoff`.
    * `save_visual_every`
        * **Type:** `integer`
        * **Description:** Save visualization outputs every N epochs.
    * `show_tol`
        * **Type:** `boolean`
        * **Description:** If `true`, displays tolerance nodes in a different color visualizations.
    * `class_weights`
        * **Type:** `float`
        * **Description:** Weight assigned to the positive class (e.g., error class) in the loss function to handle class imbalance.
    * `conf_weight`
        * **Type:** `float`
        * **Description:** Weight applied to confidence nodes in the loss function.
    * `tolerance_weight`
        * **Type:** `float`
        * **Description:** Weight applied to tolerance nodes in the loss function.
    * `box_weight`
        * **Type:** `float`
        * **Description:** Weight applied to bounding box/core nodes in the loss function.
    * `max_dist`
        * **Type:** `integer`
        * **Description:** Distance from error used to idenfity tolerance nodes. If the `max_dist` is 2 then all nodes at most 2 nodes away from an error are considered tolerance nodes.
    * `thresholds`
        * **Type:** `list of floats`
        * **Description:** A list of probability thresholds to use for evaluating model performance (e.g., for precision-recall curves).
    * `recall_targets`
        * **Type:** `list of floats`
        * **Description:** A list of target recall values for evaluating precision at specific recall levels.
    * `obj_det_error_cloud_ratios`
        * **Type:** `list of floats`
        * **Description:** Ratios used for defining error cloud sizes in object detection metrics.
    * `branch_degrees`
        * **Type:** `list of integers`
        * **Description:** A list of branch degrees (number of connected edges at a node) to consider metrics.

### `optimizer`
* **Description:** Parameters for the training optimizer.
    * `epochs`
        * **Type:** `integer`
        * **Description:** The total number of training epochs to run.
    * `lr`
        * **Type:** `float`
        * **Description:** The initial learning rate for the optimizer.

### `whole_cell`
* **Description:** Configuration specifically for whole-cell processing or evaluation.
    * `data_dir`
        * **Type:** `string`
        * **Description:** Absolute path to the base directory for whole-cell specific data.
    * `skeleton_version`
        * **Type:** `integer`
        * **Description:** The specific version of the skeletonization service used for whole-cell data.
    * `pos_enc_dim`
        * **Type:** `integer`
        * **Description:** The dimensionality of the positional encoding feature used for whole-cell data.