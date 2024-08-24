# Custom MMDetection Pipeline

This study repository contains the forked MMDetection project. It is a part of the AirDetection study. Main repo can be found [under this link](https://github.com/theATM/AirDetection). 

Here is the code used for training and testing various models. Custom configurations are
stored in the "configs/additional/" directory. This ML toolbox has a modular design. Each architecture
has its own configuration files using a very similar config structure.

### Original README [Here](README_ORIGINAL.md)

## List of all MMDetection custom configurations used
* detr/detr-dotana.py - used for the DETR experiments on Dotana
* detr/detr-rsd.py - used for the DETR experiments on RSD-GOD
* rcnn/faster-rcnn-custom.py - used for the Faster R-CNN experiments on RSD-GOD
* rcnn/faster-rcnn-custom-dotana.py - used for the Faster R-CNN experiments on Dotana
* rtm/rtmdet_tiny-rsd.py - experiments on RTM with RSD-GOD dataset
* rtm/rtmdet_tiny-dotana.py - experiments on RTM with Dotana dataset
* rtm/rtmdet_small-rsd.py - larger RTM network with RSD-GOD dataset
* rtm/rtmdet_small-dotana.py - larger RTM network with Dotana dataset
* ssd_dotana.py - used for the SSD experiments on Dotana with VGG backbone

## Usage

In order to train the models on the RSD-GOD and Dotana
datasets, the MMDetection framework requires a slight change to the COCO format. It differed from the original, as all the classes, annotations and images have identifiers started from zero. There are RSD-COCO-0 annotations aviailable to download [here](https://drive.google.com/file/d/1aypqgUDdSnJbElffF6P864MAnz0v7BLb/view?usp=sharing).

## RTMDet Results


## Model Zoo

 RTMDet | DETR | Faster R-CNN | SSD
 |  :----: | :----: | :----: | :----: |
| [T1](https://drive.google.com/file/d/1crx4ypjRHtVeqtknwi4EW6w3u_MATnTo/view?usp=sharing) [T2](https://drive.google.com/file/d/1a1c11KqA0Gt_VmFKCu_RQe-ESwam6Xp0/view?usp=sharing) <br> [T3](https://drive.google.com/file/d/1HYP-lMsq8xlZ6qG7_MmAQi5Gc33HMU8x/view?usp=sharing) [T4](https://drive.google.com/file/d/1tNolx3TwHCgiWj8NJC42Qp-pcFsvH77b/view?usp=sharing) | [D2](https://drive.google.com/file/d/1nt5jr17RP7hYRSsYByOoxuIyvWosARdf/view?usp=sharing) | [R4](https://drive.google.com/file/d/10CBfGHrepi_bTf17anL6ZzJqo2L60I3P/view?usp=sharing) | [S23](https://drive.google.com/file/d/1sAsoLrs2eh66HGHyVldOu1I2l1yBQCyV/view?usp=sharing)