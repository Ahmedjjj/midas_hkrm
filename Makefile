SHELL := /bin/bash

ENV=jellouli-env
REQUIREMENTS=requirements.txt
PYTHON=3.6
CONDA_BASE=$(shell conda info --base)
ICCLUSTER_INSTALL_DIR=/ivrldata1/students/2021-fall-sp-jellouli/.local/

MATLAB_FOLDER=$(shell dirname $(shell which matlab))/..
COMPILED_CPP_PATH=$(shell pwd)/external/edges/private

DOCKERIMAGE=jellouli_docker

create_env:
	@conda create -n $(ENV) python=$(PYTHON)

install_env_cluster:
	@source $(CONDA_BASE)/etc/profile.d/conda.sh && \
	conda activate $(ENV) && pip install -t $(ICCLUSTER_INSTALL_DIR) -r $(REQUIREMENTS)

install_env:
	@source $(CONDA_BASE)/etc/profile.d/conda.sh && \
	conda activate $(ENV) && pip3 install -r $(REQUIREMENTS)

install_matlab_engine:
	@source $(CONDA_BASE)/etc/profile.d/conda.sh && \
    conda activate $(ENV) && \
	cd $(MATLAB_FOLDER)/extern/engines/python && \
	python3 setup.py install && \
	cd -

install_matlab_dependencies:
	@matlab -batch "addpath(genpath('external/toolbox')); savepath;"
	@mex $(COMPILED_CPP_PATH)/edgesDetectMex.cpp -outdir $(COMPILED_CPP_PATH)
	@mex $(COMPILED_CPP_PATH)/edgesDetectMex.cpp -outdir $(COMPILED_CPP_PATH)
	@mex $(COMPILED_CPP_PATH)/edgesNmsMex.cpp  -outdir $(COMPILED_CPP_PATH)
	@mex $(COMPILED_CPP_PATH)/spDetectMex.cpp  -outdir $(COMPILED_CPP_PATH)
	@mex $(COMPILED_CPP_PATH)/edgeBoxesMex.cpp -outdir $(COMPILED_CPP_PATH)
	@cd external/edges && matlab -batch "addpath(pwd); savepath;" && cd -

model_save:
	@cd external/edges/models/forest && \
	matlab -batch "model = load('modelBsds'); save 'modelBsds.mat', model;" && \
	cd -

image:
	@docker build -t $(DOCKERIMAGE) .

notebook_docker: image
	@docker run --rm --network host -v $(shell pwd):/app -w /app \
	$(DOCKERIMAGE) jupyter notebook --allow-root --no-browser

