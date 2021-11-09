SHELL := /bin/bash

ENV=jellouli-env
REQUIREMENTS=requirements.txt
PYTHON=3.6
CONDA_BASE=$(shell conda info --base)
ICCLUSTER_INSTALL_DIR=/ivrldata1/students/2021-fall-sp-jellouli/.local/

DOCKERIMAGE=jellouli_docker

create_env:
	@conda create -n $(ENV) python=$(PYTHON)

install_env_cluster:
	@source $(CONDA_BASE)/etc/profile.d/conda.sh && \
	conda activate $(ENV) && pip install -t $(ICCLUSTER_INSTALL_DIR) -r $(REQUIREMENTS)

install_env:
	@source $(CONDA_BASE)/etc/profile.d/conda.sh && \
	conda activate $(ENV) && pip3 install -r $(REQUIREMENTS)

install_detectron:
	TORCH_VERSION=$(shell python -c "import torch; print('.'.join(torch.__version__.split('.')[:2]))")
	echo ${TORCH_VERSION}
	CUDA_VERSION = $(shell python -c "import torch; print(torch.__version__.split('+')[-1])")
 	pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/${CUDA_VERSION}/torch${TORCH_VERSION}/index.html"

image:
	@docker build -t $(DOCKERIMAGE) .

notebook_docker: image
	@docker run --rm --network host -v $(shell pwd):/app -w /app \
	$(DOCKERIMAGE) jupyter notebook --allow-root --no-browser

