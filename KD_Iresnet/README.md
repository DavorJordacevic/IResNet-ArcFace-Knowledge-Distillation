# Face Recognition through Template-level Knowledge Distillation

Research Paper at:

* [Arxiv](https://arxiv.org/abs/2112.05646)
* [IEEE Xplore](https://ieeexplore.ieee.org/document/9667081)

## Table of Contents 

- [Data](#data)
- [Model Training](#model-training)
- [Citing](#citing)
- [Acknowledgement](#acknowledgement)
- [License](#license)


## Data ## 

### Datasets ###
The MS1M-ArcFace dataset can be downloaded [here](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_).

For all the datasets above, please strictly follow the licence distribution.

### Model Training ###
 1. Download pretrained ARcface models
 2. Download MS1M-ArcFace dataset
 3. Set the config.rec in config/configKD.py to the dataset path
 4. Intall the requirement from requirement.txt: pip install -r requirements.txt
 5. run train_kd.py 

## Citing ##
If you use any of the code provided in this repository or the models provided, please cite the following paper:
```
@INPROCEEDINGS{huber2021maskinvariant,  
   author={Huber, Marco and Boutros, Fadi and Kirchbuchner, Florian and Damer, Naser},  
   booktitle={2021 16th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2021)},   
   title={Mask-invariant Face Recognition through Template-level Knowledge Distillation},   
   year={2021},  
   volume={},  
   number={},  
   pages={1-8},  
   doi={10.1109/FG52635.2021.9667081}
}
```

## License ##

This project is licensed under the terms of the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.
Copyright (c) 2021 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt
