# Quality-focused Active Adversarial Policy for Safe Grasping in Human-Robot Interaction (QFAAP)

<p align="center">
<img src="fig.2.jpg" width="100%"/>
<br>

The MCIGP is designed to realize grasping in large-scale dense clutter scenarios. Specifically, the first part is the Monozone View Alignment (MVA), wherein we design the dynamic monozone that can align the camera view according to different objects during grasping, thereby alleviating view boundary effects and realizing grasping in large-scale dense clutter scenarios. Then, we devise the Instance-specific Grasp Detection (ISGD) to predict and optimize grasp candidates for one specific object within the monozone, ensuring an in-depth analysis of this object. We performed over 8,000 real-world grasping experiments in different cluttered scenarios with 300 novel objects, demonstrating that MCIGP significantly outperforms seven competitive grasping methods. Notably, in a large-scale densely cluttered scene involving 100 different household goods, MCIGP pushed the grasp success rate to 84.9%. To the best of our knowledge, no previous work has demonstrated similar performance.


[arXiv](https://arxiv.org/abs/2409.06959) | [All Experimental Videos](https://youtu.be/CFlSAr5F-sI?si=UwgGsyhBBc6adlEG)

If you use this work, please cite:

```text
@inproceedings{clee2025pmsgp,
	title={Pyramid-Monozone Synergistic Grasping Policy in Dense Clutter},
	author={Chenghao, Li and Razvan, Beuran and Nak Young, Chong},
	booktitle={arXiv:2503.19397},
	year={2025}
}
```

**Contact**

Any questions or comments contact [Chenghao Li](chenghao.li@jaist.ac.jp).

## Installation

This code was developed with Python 3.8 on Ubuntu 22.04.  Python requirements can installed by:

```bash
pip install -r requirements.txt
```

## Datasets

Currently, all datasets are supported.

### Cornell Grasping Dataset

1. Download and extract the [Cornell Dataset](https://www.kaggle.com/datasets/oneoneliu/cornell-grasp). 

### OCID Grasping Dataset

1. Download and extract the [OCID Dataset](https://files.icg.tugraz.at/d/777515d0f6e74ed183c2/).

### Jacquard Grasping Dataset

1. Download and extract the [Jacquard Dataset](https://jacquard.liris.cnrs.fr/).


## Pre-trained Grasping Models

All pre-trained grasping models for GG-CNN, GG-CNN2, GR-Convnet, and others can be downloaded from [here](https://drive.google.com/drive/folders/1Yos_urL8h1A_kFrnu2y2xCD7uGeuTDGJ?usp=sharing).

## Pre-trained AQP

All AQP trained by different grasping models and datasets are available at the 'AQP examples' file.

## Pre-trained Hand Segmentation Models
All pre-trained Hand Segmentation models can be downloaded from [here](https://github.com/Unibas3D/Upper-Limb-Segmentation) or [here](https://drive.google.com/drive/folders/1yd6nKRaRFG7-vIRMzp3JM11slkIvzUCr?usp=sharing).

## Training/Evaluation

Training for AQP is done by the `AQP_training.py`.  Training for Grasping model is done by the `train_grasping_network.py`. 
And the evaluation process is followed by the training.

## Predicting
1. The offline and realtime prediction of QFAAP is done by the `QFAAP_offline.py` and `QFAAP_realtime.py`.
2. For the deployment of real-time hand segmentation, please refer to this repository [https://github.com/Unibas3D/Upper-Limb-Segmentationp](https://github.com/Unibas3D/Upper-Limb-Segmentation) 

<p align="center">
<img src="fig.6.jpg" width="100%"/>
<br>


## Running on a Robot

1. Please reference this repository  [https://github.com/dougsm/ggcnn_kinova_grasping](https://github.com/dougsm/ggcnn_kinova_grasping)
2. Or [https://github.com/clee-jaist/MCIGP](https://github.com/clee-jaist/MCIGP)

<p align="center">
<img src="fig.7.jpg" width="100%"/>
<br>


All grasping videos are recoded at: https://www.youtube.com/@chenghaoli4532/playlists
