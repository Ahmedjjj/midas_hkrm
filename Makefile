SHELL := /bin/bash

ENV=jellouli-env
REQUIREMENTS=requirements.txt
PYTHON=3.9
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
	@source $(CONDA_BASE)/etc/profile.d/conda.sh && \
	conda activate $(ENV) && \
	python -c "import torch, os ; \
 			   TORCH_VERSION = '.'.join(torch.__version__.split('.')[:2]);\
 			    CUDA_VERSION = torch.__version__.split('+')[-1]; \
 			    os.system( \
	f'pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/{CUDA_VERSION}/torch{TORCH_VERSION}/index.html')"

image:
	@docker build -t $(DOCKERIMAGE) .

notebook_docker: image
	@docker run --rm --network host -v $(shell pwd):/app -w /app \
	$(DOCKERIMAGE) jupyter notebook --allow-root --no-browser

