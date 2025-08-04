# DIP-STER
Deep Image Priors for Space Time Environment Reconstruction (DIP-STER) is a nueral network that uses Deep Image Priors for machine learning using an architecture involving manifold learning and convolution nueral networks in order to determine the resolve a 4D(3D + time) series of electron tomography data during _in situ_ experiments. It is build upon the previous work of [Time Dependent Deep Image Priors](https://github.com/jaejun-yoo/TDDIP). 

### Table of Contents
- [Installation](#1-installation)
- [Usage](#2-usage)
- [License](#3-license)
- [Citation](#4-citations)

## 2. Installation 
This package can be installed through conda using the following command.

```bash
    conda create -n <new_env>
    conda activate

    conda install tomosipo pytorch torchvision torchaudio cudatoolkit=<x.x> pytorch-cuda=<x.x> -c aahendriksen -c astra-toolbox -c pytorch -c nvidia
    pip install -e .
```
Current installation has been tested with pytorch 2.0.1  and cudatoolkit/pytorch cuda 11.7.


## 3. Usage 
[View the tutorial notebook on GitHub](https://github.com/Tcraig088/dipster/blob/main/scripts/tutorial.ipynb)


## 4. License 

This code is licensed under GNU general public license version 3.0.

## 5. Citations
This section is to be completed 

## 6. Contributors
EMAT: Timothy Craig - tim.craig@uantwerpen.be

## 1.6. Contributors
EMAT: Timothy Craig - tim.craig@uantwerpen.be

## 1.7 Debug Notice


