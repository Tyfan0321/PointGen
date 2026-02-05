## Promptable Generative 3D Dense Matching

### Installation

Please use the following command for installation.

```bash
# It is recommended to create a new environment
conda create -n pointgen python==3.7
conda activate pointgen

# Install packages and other dependencies
pip install -r requirements.txt

# If you are using CUDA 11.2 or newer, you can install `torch==1.7.1+cu110` or `torch=1.9.0+cu111`
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Install pytorch3d (feel free to download it to other directories)
conda install openblas-devel -c anaconda
wget https://github.com/facebookresearch/pytorch3d/archive/refs/tags/v0.6.2.zip
mv v0.6.2.zip pytorch3d-0.6.2.zip
unzip pytorch3d-0.6.2.zip
cd pytorch3d-0.6.2
pip install -e . 
cd ..

# Install MinkowskiEngine (feel free to download it to other directories)
git clone https://github.com/NVIDIA/MinkowskiEngine
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas

# Install Accelerate for distributed training
pip install accelerate

# Download pre-trained weights from release v1.0.0
```

Code has been tested with Ubuntu 20.04, GCC 9.4.0, Python 3.7, PyTorch 1.9.0, CUDA 11.2 and PyTorch3D 0.6.2.

### Project Structure

```
PointGen/
├── src/                       # New framework code
│   ├── models/                # Model implementations
│   │   ├── generative/        # Generative model code
│   │   └── kpconv/            # KPConv model code
│   ├── engine/                # Training engine
│   │   ├── base_trainer.py    # Base trainer class
│   │   ├── diffusion_trainer.py # Diffusion model trainer
│   │   ├── data_processor.py  # Data processor
│   │   └── evaluator.py       # Model evaluator
│   ├── scripts/               # Scripts
│   │   └── train.py           # Main training script
│   ├── utils/                 # Utility functions
│   │   └── config_utils.py    # Config utilities
│   └── data/                  # Data processing
│       └── dataset_factory.py # Dataset factory
├── config/                    # Configuration files
│   ├── config.yaml            # Main configuration file
│   ├── 3dmatch_fm.json        # 3DMatch config
│   ├── kitti_fm.json          # KITTI config
│   └── kitti_gen.json         # KITTI generative config
├── data/                      # Data lists and processing scripts
├── .vscode/                   # VS Code configuration
├── .gitignore                 # Git ignore file
├── LICENSE                    # License file
├── README.md                  # This file
└── requirements.txt           # Dependencies
```

### Data Preparation

#### 3DMatch and 3DLoMatch
The dataset can be downloaded from [PREDATOR](https://github.com/prs-eth/OverlapPredator) (by running the following commands):
```bash
wget --no-check-certificate --show-progress https://share.phys.ethz.ch/~gsg/pairwise_reg/3dmatch.zip
unzip 3dmatch.zip
```
The data should be organized as follows:
- `3dmatch`
    - `train`
        - `7-scenes-chess`
            - `fragments`
                - `cloud_bin_*.ply`
                - ...
            - `poses`
                - `cloud_bin_*.txt`
                - ...
        - ...
    - `test`
        - `7-scenes-redkitchen`
            - `fragments`
                - `cloud_bin_*.ply`
                - ...
            - `poses`
                - `cloud_bin_*.txt`
                - ...
        - ...

#### KITTI
Download the KITTI Odometry dataset from [KITTI website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).

### Configuration

Modify the configuration file `config/config.yaml` to set your dataset paths and other parameters:

```yaml
# Data configuration
data:
  dataset_type: "kitti"  # Options: "kitti", "3dmatch"
  root: "/path/to/your/dataset"
  data_list: "./data/kitti_list/"
  npoints: 30000
  voxel_size: 0.3
  augment: 1.0
```

### Training

#### Single GPU Training
```bash
python src/scripts/train.py
```

#### Multi-GPU Training
Using Accelerate for distributed training:

```bash
# Using all available GPUs
accelerate launch src/scripts/train.py

# Using specific GPUs
CUDA_VISIBLE_DEVICES=0,1 accelerate launch src/scripts/train.py

# With specific number of processes
accelerate launch --num_processes=2 src/scripts/train.py
```

#### Resume Training
```bash
# Single GPU
python src/scripts/train.py --resume ./result/checkpoint-epoch-10

# Multi-GPU
accelerate launch src/scripts/train.py --resume ./result/checkpoint-epoch-10
```

### Evaluation

The evaluation is automatically performed during training at regular intervals. You can also run evaluation separately by loading a trained model:

```bash
# Single GPU evaluation
python src/scripts/train.py --resume ./result/checkpoint-epoch-30 --eval_only

# Multi-GPU evaluation
accelerate launch src/scripts/train.py --resume ./result/checkpoint-epoch-30 --eval_only
```

### Key Features

1. **Modular Architecture**: Clean separation between model, data processing, training logic, and evaluation.

2. **Distributed Training**: Support for multi-GPU training using Accelerate.

3. **Configuration Management**: Centralized YAML configuration with snapshot saving for experiment reproducibility.

4. **Dataset Factory**: Dynamic dataset creation based on configuration.

5. **Diffusion Model Support**: Integration with FlowMatchEulerDiscreteScheduler for point cloud generation.

6. **Comprehensive Evaluation**: Detailed metrics including distance error, RRE, RTE, and overlap region metrics.

### Coding Example

- **Model Implementation**: `src/models/generative/transformer_regtr.py` implements the RegTrGenerative model.
- **Training Logic**: `src/engine/diffusion_trainer.py` implements the diffusion model training logic.
- **Data Processing**: `src/engine/data_processor.py` handles data preparation and loss calculation.
- **Evaluation**: `src/engine/evaluator.py` computes evaluation metrics.
- **Main Script**: `src/scripts/train.py` is the entry point for training and evaluation.

