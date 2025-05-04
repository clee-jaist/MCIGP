# Monozone-centric Instance Grasping Policy in Large-scale Dense Clutter (MCIGP)

<p align="center">
<img src="fig.2.jpg" width="100%"/>
<br>

The MCIGP is designed to realize grasping in large-scale dense clutter scenarios. Specifically, the first part is the Monozone View Alignment (MVA), wherein we design the dynamic monozone that can align the camera view according to different objects during grasping, thereby alleviating view boundary effects and realizing grasping in large-scale dense clutter scenarios. Then, we devise the Instance-specific Grasp Detection (ISGD) to predict and optimize grasp candidates for one specific object within the monozone, ensuring an in-depth analysis of this object.


[arXiv](https://arxiv.org/abs/2409.06959) | [All Experimental Videos](https://youtu.be/CFlSAr5F-sI?si=UwgGsyhBBc6adlEG)

If you use this work, please cite:

```text
@inproceedings{clee2025pmsgp,
	title={Pyramid-Monozone Synergistic Grasping Policy in Dense Clutter},
	author={Chenghao, Li and Nak Young, Chong},
	booktitle={https://arxiv.org/abs/2409.06959},
	year={2024}
}
```

**Contact**

Any questions or comments contact [Chenghao Li](chenghao.li@jaist.ac.jp).

## Installation

This code was developed with Python 3.7. Requirements can installed by:

```bash
pip install -r requirements.txt
```

## Hardware

The code was deployed by UFactory 850/Xarm5 Robot and Intel Realsense D435i.

1. UFactory Robot API: [https://github.com/xArm-Developer/xArm-Python-SDK](https://github.com/xArm-Developer/xArm-Python-SDK).
2. Intel Realsense API: [https://github.com/IntelRealSense/librealsense](https://github.com/IntelRealSense/librealsense).

## Datasets

### Cornell Grasping Dataset

1. Download and extract the [Cornell Dataset](https://www.kaggle.com/datasets/oneoneliu/cornell-grasp). 

### OCID Grasping Dataset

1. Download and extract the [OCID Dataset](https://files.icg.tugraz.at/d/777515d0f6e74ed183c2/).


## Pre-trained Grasping Models

Has been included in this code as 'GRconvnet_RGBD_epoch_40_iou_0.52'.

## Pre-trained SAM model
Please refer to this repository [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) 

## Training and predicting for original grasping model

Training is done by the `train_network`, predicting is done by `grasp detection`.

## Running on a Robot

1. Real robot grasping is done by `MCIGP grasping`
2. Note: Please use your own hand-eye calibration results when deploying.

<p align="center">
<img src="fig.6.jpg" width="100%"/>
<br>

<p align="center">
<img src="fig.7.jpg" width="100%"/>
<br>
