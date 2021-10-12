#!/bin/bash
conda create -n "jellouli-env" python=3.3.0 
conda init bash
conda activate jellouli-env
pip install -r requirements.txt
