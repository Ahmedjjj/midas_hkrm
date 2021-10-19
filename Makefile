SHELL := /bin/bash

ENV=jellouli-env
REQUIREMENTS=requirements.txt
PYTHON=3.6
DOCKERIMAGE=jellouli_docker
CONDA_BASE=$(shell conda info --base)
ICCLUSTER_INSTALL_DIR=/ivrldata1/students/2021-fall-sp-jellouli/.local/
MATLAB_FOLDER=$(shell dirname $(shell which matlab))/..

create_env:
	@conda create -n $(ENV) python=$(PYTHON)

install_env:
	@source $(CONDA_BASE)/etc/profile.d/conda.sh && \
	conda activate $(ENV) && pip install -t $(ICCLUSTER_INSTALL_DIR) -r $(REQUIREMENTS)

install_matlab_engine:
	source $(CONDA_BASE)/etc/profile.d/conda.sh && \
	conda activate $(ENV) && python3 $(MATLAB_FOLDER)/extern/engines/python/setup.py install

image:
	@docker build -t $(DOCKERIMAGE) .

notebook: image
	@docker run --rm --network host -v $(shell pwd):/app -w /app \
	$(DOCKERIMAGE) jupyter notebook --allow-root --no-browser

