# DIP-STER
Deep Image Priors for Space Time Environment Reconstruction (DIP-STER) is a nueral network that uses Deep Image Priors for machine learning using an architecture involving manifold learning and convolution nueral networks in order to determine the resolve a 4D(3D + time) series of electron tomography data during _in situ_ experiments. It is build upon the previous work of [Time Dependent Deep Image Priors](https://github.com/jaejun-yoo/TDDIP). 

### Table of Contents
- [DIP-STER](#dip-ster)
    - [Table of Contents](#table-of-contents)
  - [1. Installation](#1-installation)
    - [Prerequisites:](#prerequisites)
    - [Installation](#installation)
  - [2. Usage](#2-usage)
  - [3. License](#3-license)
  - [4. Citations](#4-citations)
  - [5. Contributors](#5-contributors)

## 1. Installation
### Prerequisites:
- Up to date graphic card compatible with CUDA (see [NVIDIA driver page](https://www.nvidia.com/en-us/drivers/) and [CUDA installation page](https://docs.nvidia.com/cuda/index.html))
- Miniconda or Anaconda (see [Miniconda page](https://www.anaconda.com/docs/getting-started/miniconda/install))

This package also requires an additional modules to use the full extent of this code.
- [tomobase](https://github.com/Tcraig088/tomobase/tree/mvc_branch)

### Installation
In a dedicated folder, clone this repository:
```bash
    git clone -b main https://github.com/Tcraig088/dipster.git
```

To prevent any conflict with your current python installation, it is recommended to create a dedicated python environment:
```bash
    conda create -n <new_env> python=3.11
    conda activate <new_env>

    # CUDA 12.4
    conda install conda cuda-toolkit pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 kornia 'wandb==0.18.7' stackview cupy pyqt hyperspy-base mrcfile astra-toolbox==2.4 numpy -c pytorch -c nvidia -c astra-toolbox -c conda-forge
    pip install git+https://github.com/Tcraig088/tomobase@recovery_branch qtpy blinker coolname "ray[data,train,tune,serve]"
    pip install --no-deps --no-build-isolation git+https://github.com/ahendriksen/tomosipo.git
    pip install -e ./dipster/  # Name of the folder where dispster was cloned

    # CUDA 13.0
    conda install conda cuda-toolkit kornia 'wandb==0.18.7' stackview cupy pyqt hyperspy-base mrcfile astra-toolbox==2.4 numpy -c astra-toolbox -c conda-forge
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
    pip install git+https://github.com/Tcraig088/tomobase@recovery_branch qtpy blinker coolname "ray[data,train,tune,serve]"
    pip install --no-deps --no-build-isolation git+https://github.com/ahendriksen/tomosipo.git
    pip install -e ./dipster/  # Name of the folder where dispster was cloned
```

This installation was tested on Windows 11 and Ubuntu-24.04 with WSL for Win11. Both used CUDA version 12.4.
Newer version of pytorch compatible with CUDA 13.0 also seems to work for newer graphic card.
Check the list_packages.txt or environment.yml file as a reference if you have some issues.

Installation with pytorch 2.0.1 and cudatoolkit/pytorch cuda 11.7 was also tested.

## 2. Usage 
[View the tutorial notebook on GitHub](https://github.com/Tcraig088/dipster/blob/main/scripts/tutorial.ipynb)

## 3. License 
This code is licensed under GNU general public license version 3.0.

## 4. Citations
**This section is to be completed** 

## 5. Contributors
EMAT: Timothy Craig - tim.craig@uantwerpen.be
      Adrien Moncomble - adrien.moncomble@uantwerpen.be
