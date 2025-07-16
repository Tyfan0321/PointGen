## Promptable Generative 3D Dense Matching

### Installation

Please use the following command for installation.

```bash
# It is recommended to create a new environment
conda create -n cast python==3.7
conda activate cast

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

# Download pre-trained weights from release v1.0.0
```

Code has been tested with Ubuntu 20.04, GCC 9.4.0, Python 3.7, PyTorch 1.9.0, CUDA 11.2 and PyTorch3D 0.6.2.



### 3DMatch and 3DLoMatch

#### Data preparation

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

#### Training
After modifying the ```data.root``` item to your dataset path (```/remote-home/ums_huangrenlang/3dmatch```) in ```./config/3dmatch.json```, you can use the following command for training.
```bash
python train_a1.py --mode train --config ./config/3dmatch.json
```

#### Testing
After modifying the ```data.root``` item to your dataset path in ```./config/3dmatch.json```, you can use the following command for testing.
```bash
python train_a1.py --mode train --config ./config/3dmatch.json --load_pretrained regtr1-epoch-30
```


### KITTI odometry

#### Data preparation

Download the data from the [KITTI official website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). The data should be organized as follows:
- `KITTI`
    - `velodyne` (point clouds)
        - `sequences`
            - `00`
                - `velodyne`
                    - `000000.bin`
                    - ...
            - ...
    - `results` (poses)
        - `00.txt`
        - ...
    - `sequences` (sensor calibration and time stamps)
        - `00`
            - `calib.txt`
            - `times.txt`
        - ...

Please note that we have already generated the information of pairwise point clouds via ``./data/gen_kitti_data.py``, which is stored in ``./data/kitti_list``. Feel free to use it directly or re-generate the information by yourselves.

#### Training
After modifying the ```data.root``` item to your dataset path in ```./config/kitti.json```, you can use the following command for training.
```bash
python trainval.py --mode train --config ./config/kitti.json
```

#### Testing
After modifying the ```data.root``` item to your dataset path in ```./config/kitti.json```, you can use the following command for testing.
```bash
python trainval.py --mode test --config ./config/kitti.json --load_pretrained cast-epoch-40
```
