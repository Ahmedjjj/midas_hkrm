
## Introduction
This is the code for my semester project at the Image and Visual Representation Lab (IVRL) at EPFL. The idea is to explore the usage of object detection for monocular depth estimation. Please see the file (TODO INSERT REPORT NAME) for details.

## Cloning the repository
Please use `git clone --recursive` instead of `git clone` so as to clone the used submodule.
Otherwise run `git submodule init && git submodule update`.

## Requirements
You may edit the conda environment name in the Makefile then run `make env` in order to install the required dependencies.  
The code of our [fork](https://github.com/Ahmedjjj/MiDaS) of [MiDaS](https://github.com/isl-org/MiDaS) should be on `PATH` or `PYTHONPATH`.  
Otherwise, the file requirements.txt specifies all needed packages to run the project.  
Additionally, `torch` and `torchvision` are required. We omit these from the requirements because we faced many issues with CUDA version dependencies.  
[light-the-torch](https://github.com/pmeier/light-the-torch) is a a great tool that installs the right version automatically.  
Finally, `detectron2` needs to be installed: [instructions](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md).  
Note that the `make env` installs the right version of `torch`, `torchvision` and `detectron2`.

## Package structure:
All the code is under the **midas_hkrm** package and is documented. The stucture of the sub-packages is the following:
- `datasets`: these are the torch Dataset abstractions for all the datasets used in the project
- `objects`: contains the code for the modified HKRM model that we train for object detection, as well as an `ObjectDetector` abstraction used in the depth model
- `optim`: contains the loss function, as well as useful Trainer and Tester abstractions
- `depth`: containes the modified `MidasHKRMNet`
- `utils`: contains common util functions
- `zero_shot`: contains the code we use for zero-shot evaluation. Namely the different evaluation criteria as well as an `Evaluator` abstraction.

## Top level scripts
- `train_hkrm.py`: script that we used to train the modified hkrm model
- `train_midas_hkrm.py` : script that we used to train one of the MidasHKRM models
- `eval_midas.py`: script to run evaluations on MiDas 2.1. The script takes the following arguments:
  ``` text
     usage: eval_midas.py [-h] [--cpu] [--nyu] [--tum] [--eth] [--save_path SAVE_PATH]
     optional arguments:
      -h, --help            show this help message and exit
      --cpu  use the cpu instead of the gpu
      --nyu  eval on NYUv2 test Set
      --tum  eval on TUM dynamic subset (only for static camera)
      --eth  eval on ETH3D
      --save_path SAVE_PATH save results as dict
     ```
 - `eval_midas_hkrm.py`: script to run evaluations of MidasHKRM. The script takes the following arguments:
 - `eval_hkrm.py`: script to run COCO 2017 validation set evaluation on the modified HKRM model. The script takes the following arguments


## Reproducilbility (Only on the cluster)
For the HKRM COCO 2017 test set results:
``` bash
    cd /runai-ivrl-scratch/students/2021-fall-sp-jellouli/midas_hkrm
    conda activate jellouli-env
    python eval_hkrm.py --model /runai-ivrl-scratch/students/2021-fall-sp-jellouli/output/model_(TODO).pth
```
For the reported results on Midas 2.1:
``` bash
    cd /runai-ivrl-scratch/students/2021-fall-sp-jellouli/midas_hkrm
    conda activate jellouli-env
    python eval_midas.py --nyu --tum --eth
```
For the reported MidasHKRM results:
TODO
For the reported MidasBASE results:
TODO
## Acknowledgments
This code is based on:
- MiDas: [paper](https://arxiv.org/abs/1907.01341), [code](https://github.com/isl-org/MiDaS)
- HKRM: [paper](https://arxiv.org/abs/1810.12681), [code](https://github.com/chanyn/HKRM)
- Detectron2: [code](https://github.com/facebookresearch/detectron2)


