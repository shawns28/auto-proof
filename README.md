# AutoProof

Automated Proofreading Detection for EM Connectomics

The paper is currently in progress...

## Requirements

All training requires a GPU. Experiments were run using [Slurm](https://slurm.schedmd.com/documentation.html) for GPU allocation on a range of machines ranging from Nvidia 1080 to A100. A single epoch with the default config takes roughly 20-30 min.  

## Set up

### Conda
We used [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to create virtual environments. Below is an example creating an environment in conda. Run the below commands to create and activate the environment, install pip and install all the packages in `requirements.txt`. We name the default env name: `auto_env`.

1. `conda create --name auto_env`
1. `conda activate auto_env`
1. `conda install pip`
1. `pip install --upgrade .`

### Data

Contact sven.dorkenwald@alleninstitute.org for a copy of the preprocessed features and model checkpoints.

### Config

There are 3 default configs in `auto_proof/configs/`:
- For training: `base_config.json`, documentation: `base_config.md`
- For client and mat versions: `client_config.json`, documentation: `client_config.md` 
- For pre-processing data: `data_config.json`, documentation: `data_config.md`

Make sure to set your [cave token](https://tutorial.microns-explorer.org/quickstart_notebooks/00_cave_quickstart.html) in `client_config.json`.

Set the correct paths to these configs in `auto_proof/code/pre/data_utils` which handles loading in the configs in various other files.

### Neptune

We use [Neptune](https://neptune.ai/) for experiment tracking. It allows you to see real-time logs, cpu/gpu utilization, save the state of specified files, compare experiments and view artificats/plots for each experiment easily. 

Make sure to set your [Neptune api token](https://docs.neptune.ai/api_token/) in `auto_proof/code/main/main.py`.

## Training

Run `auto_proof/code/main/main.py` if already on a gpu or one of the sbatch scripts like `sbatch/auto_proof/code/main/main_gpu64.sh` to start training.

## Visualization

Visualizations are created using [PyVista](https://pyvista.org/). Reference `auto_proof/code/visualize.py` for further information on how we implement our visualizations. Our visualizations are stored as static html files that can be viewed interactively in a browser.

To run visualization for an example root, run `python -m auto_proof.code.visualize`. 

## Evaluation

Evaluation is done at the node-level and object-level. These metrics are calculated during the validation epoch in training.

Node-level metrics are effectively deprecated and should only be referenced as a sanity check.

Object-level metrics are calculated by treating errors as connected-components. There are various pre-processing steps done to the valication set to pick roots that are good candidates for this evaluation. More information on this can be found in `auto_proof/code/pre/train_test_split.py` and `auto_proof/code/object_detection.py`.

To run object detection on a set of roots, run `python -m auto_proof.code.object_detection`.

## Whole Cell Prediction

Whole cell prediction will fetch the full skeleton for a given root, create all the features, run inference on overlapping cutouts and finally produce a prediction for the entire root. Reference `auto_proof/code/whole_cell_prediction.py` for implemenation information.

To run whole cell prediction for an example root, run `python -m auto_proof.code.whole_cell_prediction`.

## Presentation

The [Google Slides Link](https://docs.google.com/presentation/d/1TfCGLKy_9qY4UNYAZUvIV4orKXNUV_N0i5nORunV6TA/edit?usp=sharing)

`auto_proof/data/figures/presentation/` contains the final plots and visualization html files. The presentation should have links to these files in the speaker notes.

`auto_proof/data/ckpt/` contains the model checkpoints.
AUT-330 is the baseline experiment used for results.
AUT-331 is trained without SegCLR.
Due to an update with the object detection logic, the plots in those folders are outdated and the ones in the presentation figures folder should be used instead.

## Pre-Processing

NOTE: When running slurm sbatch scripts the logs directory must exist before executing the script. The directory will always be at `auto_proof/data/logs/`. If using a smaller amount of roots you could run this all on a srun session. Example srun command: `srun -c64 --mem=32000 -t3:00:00 -p celltypes --pty bash`.

NOTE: The data_config uses 32 processes as the default but if you have access to more processes on your machine, you can change this in the data_config and the scripts. The num chunks is set to 24 for default which assumes that you have access to 24 separate machines but this can be changed/set to 1.

To start pre-processing you need the initial splitlog of edits which should contain a feather file with [date, timestamp, mean_coord_x, mean_coord_y, mean_coord_z, operation_id] and a npy file with [operation_id, sink_supervoxel_id, source_supervoxel_id]. You also need a csv representing the list of fully proofread (clean axons) neurons which can be downloaded from the [proofreading_status_and_strategy](https://tutorial.microns-explorer.org/proofreading.html) cave table. Set these paths accordingly in the data_config. Set your [cave token](https://tutorial.microns-explorer.org/quickstart_notebooks/00_cave_quickstart.html) in client_config as well.

1. Run `sbatch auto_proof/code/pre/process_raw_edits.sh`. This will process the initial splitlog, create intermediate dictionaries and finally generate an initial list of root IDs and a mapping from root ID to representative edit coordinates. Estimated time: 3 hours.
1. Run `sbatch auto_proof/code/pre/process_proofread.sh`. This will process the initial fully proofread roots at the given `mat_versions`, add it to the root list, and add copies based on the `copy_count` for those proofread roots. Estimated time: 2 sec.
1. Run `sbatch auto_proof/code/pre/generate_skels.sh`. This will queue up skeletons to be added to the Cave skeleton cache using a low priority queue to avoid interfering with general users of the skeleton cache. Estimated time: 30 sec per new root (atleast 1 day but depends on number of processes on the bulk generate skeletons api). If getting rate limited contact the Allen Connectomics team or sven.dorkenwald@alleninstitute.org for an exception.
1. Run `sbatch auto_proof/code/pre/post_generate_skels.sh`. This will check which of the queued skeletons were added to the skeleton cache. Ensures that we aren't generating fresh skeletons on further steps. Estimated time: 30 min.
1. Run `sbatch auto_proof/code/pre/process_features.sh`. This will pull the skeletons from the skeleton cache and store each skeleton in a HDF5 file. It will also generate the positional encodings and save those to the same file. `cutoff` represents how many nodes will be saved from the original skeleton. `box_cutoff` represents the core nodes that we use for higher confidence prediction/evaluation. `pos_enc_dim` represents the number of dimenions we are calculating for the spectral embedding. The current spectral embedding is based off of work from https://github.com/PKU-ML/LaplacianCanonization. Estimated time: 1 day.
1. The next two steps can be run in parallel.
    1. Run `sbatch auto_proof/code/pre/process_labels.sh`. This will create the roots_at (future roots), labels, confidences and distanes to error for each root. `labels_type` represents the type of labels, 'ignore_inbetween' or 'ignore_inbetween_and_edge'. `ignore_edge_ccs` when true ignores errors at the edge of branches. Estimated time: 3 hours.
    1. Run `sbatch auto_proof/code/pre/process_segclr.sh`. This will create the roots_at for the appropriate SegCLR version and then get the associated SegCLR embedding from Google Cloud and assign the appropriate embeddings to each node in the original cutout. Estimated time: 1.5 days.
1. Run `sbatch auto_proof/code/pre/train_test_split.sh`. This will split the dataset into train/val/test. It will also generate the evaluation set for each split as well. Estimated time: 30 min.