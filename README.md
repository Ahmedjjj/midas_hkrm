
## Introduction
This is the code for my semester project at the Image and Visual Representation Lab (IVRL) at EPFL. The idea is to explore the usage of object detection for monocular depth estimation. Please see the file report_jellouli.pdf for details.

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
Note that `make env` installs the right version of `torch`, `torchvision` and `detectron2`.

## Package structure:
All the code is under the **midas_hkrm** package and is documented. The stucture of the sub-packages is the following:
- `datasets`: these are the torch Dataset abstractions for all the datasets used in the project
- `objects`: contains the code for the modified HKRM model that we train for object detection, as well as an `ObjectDetector` abstraction used in the depth model
- `optim`: contains the loss function, as well as useful Trainer and Tester abstractions
- `depth`: containes the modified `MidasHKRMNet`
- `utils`: contains common util functions
- `zero_shot`: contains the code we use for zero-shot evaluation. Namely the different evaluation criteria as well as an `Evaluator` abstraction.

## Top level scripts
- `train_hkrm.py`: script that we used to train the modified HKRM model
- `train_midas_hkrm.py` : script that we used to train one of the MidasHKRM models
- `optimize_hyperparams.py`: script that we used for the hyper-parameter search
- `eval_midas.py`: script to run evaluations on MiDas 2.1. The script takes the following arguments:
  ``` text
     usage: eval_midas.py [-h] [--cpu] [--nyu] [--tum] [--eth] [--save_path SAVE_PATH]
     optional arguments:
      -h, --help            show this help message and exit
      --cpu  # use the cpu instead of the gpu
      --nyu  # eval on NYUv2 test Set
      --tum  # eval on TUM dynamic subset (only for static camera)
      --eth  # eval on ETH3D
      --save_path SAVE_PATH save results as dict
     ```
 - `eval_midas_hkrm.py`: script to run evaluations of MidasHKRM. The script takes the following arguments:
 ```text
 usage: eval_midas_hkrm.py [-h] --object_weights OBJECT_WEIGHTS [--max_objects MAX_OBJECTS] [--detection_threshold DETECTION_THRESHOLD] --states [STATES ...] [--cpu] [--nyu] [--tum]
                          [--eth] [--base] [--test_set] [--save_path SAVE_PATH]

Eval a MidasHKRM model

optional arguments:
  -h, --help            show this help message and exit
  --object_weights OBJECT_WEIGHTS, -o OBJECT_WEIGHTS
  --max_objects MAX_OBJECTS, -m MAX_OBJECTS
  --detection_threshold DETECTION_THRESHOLD, -t DETECTION_THRESHOLD
  --states [STATES ...], -s [STATES ...] # state files to eval on
  --cpu
  --nyu
  --tum
  --eth 
  --base # use MidasBASE
  --test_set # eval on test set
  --save_path SAVE_PATH
```
 - `eval_hkrm.py`: script to run COCO 2017 validation set evaluation on the modified HKRM model. The script takes the following arguments:
 ``` text
    usage: eval_hkrm.py [-h] model_state

    Eval a HKRM model

    positional arguments:
      model_state

    optional arguments:
      -h, --help   show this help message and exit
  ```

## Reproducibility (Only on the cluster)
For the HKRM COCO 2017 test set results:
``` bash
    cd /runai-ivrl-scratch/students/2021-fall-sp-jellouli/midas_hkrm
    conda activate jellouli-env
    PYTHONPATH=$(pwd)/external/MiDaS:$PYTHONPATH DETECTRON2_DATASETS=$(pwd)/data/datasets python3 eval_hkrm.py /runai-ivrl-scratch/students/2021-fall-sp-jellouli/output/model_0254999.pth
```
For the reported results on Midas 2.1:
``` bash
    cd /runai-ivrl-scratch/students/2021-fall-sp-jellouli/midas_hkrm
    conda activate jellouli-env
    PYTHONPATH=$(pwd)/external/MiDaS:$PYTHONPATH ZERO_SHOT_DATASETS=/runai-ivrl-scratch/students/2021-fall-sp-jellouli/zero_shot_datasets python eval_midas.py --nyu
```
For the reported MidasRandom results:
``` bash
    cd /runai-ivrl-scratch/students/2021-fall-sp-jellouli/midas_hkrm
    conda activate jellouli-env
    PYTHONPATH=$(pwd)/external/MiDaS:$PYTHONPATH ZERO_SHOT_DATASETS=/runai-ivrl-scratch/students/2021-fall-sp-jellouli/zero_shot_datasets python eval_midas.py --nyu -m 20 -t 0.4 -s /runai-ivrl-scratch/students/2021-fall-sp-jellouli/out_midas_hkrm/model_70000.tar -o /runai-ivrl-scratch/students/2021-fall-sp-jellouli/output/model_final.pth
```

For the reported MidasHKRMV2 results:
``` bash
    cd /runai-ivrl-scratch/students/2021-fall-sp-jellouli/midas_hkrm
    conda activate jellouli-env
    PYTHONPATH=$(pwd)/external/MiDaS:$PYTHONPATH ZERO_SHOT_DATASETS=/runai-ivrl-scratch/students/2021-fall-sp-jellouli/zero_shot_datasets python eval_midas.py --nyu -m 20 -t 0.4 -s /runai-ivrl-scratch/students/2021-fall-sp-jellouli/out_midas_hkrm_v2/model_69999.tar -o /runai-ivrl-scratch/students/2021-fall-sp-jellouli/output/model_final.pth
```
For the reported MidasHKRMV3 results:
```bash
    cd /runai-ivrl-scratch/students/2021-fall-sp-jellouli/midas_hkrm
    conda activate jellouli-env
    PYTHONPATH=$(pwd)/external/MiDaS:$PYTHONPATH ZERO_SHOT_DATASETS=/runai-ivrl-scratch/students/2021-fall-sp-jellouli/zero_shot_datasets python eval_midas.py --nyu -m 16 -t 0.3 -s /runai-ivrl-scratch/students/2021-fall-sp-jellouli/out_midas_hkrm_v3/model_69999.tar -o /runai-ivrl-scratch/students/2021-fall-sp-jellouli/output/model_final.pth
```
    
For the reported MidasHKRMV4 results:
```bash
    cd /runai-ivrl-scratch/students/2021-fall-sp-jellouli/midas_hkrm
    conda activate jellouli-env
    PYTHONPATH=$(pwd)/external/MiDaS:$PYTHONPATH ZERO_SHOT_DATASETS=/runai-ivrl-scratch/students/2021-fall-sp-jellouli/zero_shot_datasets python eval_midas.py --     nyu -m 15 -t 0.5 -s /runai-ivrl-scratch/students/2021-fall-sp-jellouli/out_midas_hkrm_v4/model_69999.tar -o /runai-ivrl-scratch/students/2021-fall-sp-             jellouli/output/model_final.pth
```
For the reported MidasBASE results:
```bash
    cd /runai-ivrl-scratch/students/2021-fall-sp-jellouli/midas_hkrm
    conda activate jellouli-env
    PYTHONPATH=$(pwd)/external/MiDaS:$PYTHONPATH ZERO_SHOT_DATASETS=/runai-ivrl-scratch/students/2021-fall-sp-jellouli/zero_shot_datasets python eval_midas.py --nyu --base -m 15 -t 0.5 -s /runai-ivrl-scratch/students/2021-fall-sp-jellouli/out_midas_obj_baseline/model_69999.tar
```
## Acknowledgments
This code is based on:
- MiDaS: [paper](https://arxiv.org/abs/1907.01341), [code](https://github.com/isl-org/MiDaS)
- HKRM: [paper](https://arxiv.org/abs/1810.12681), [code](https://github.com/chanyn/HKRM)
- Detectron2: [code](https://github.com/facebookresearch/detectron2)


