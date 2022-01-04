SHELL := /bin/bash

ENV=jellouli-env
REQUIREMENTS=requirements.txt
PYTHON=3.9
CONDA_BASE=$(shell conda info --base)

create_env:
	@conda create -n $(ENV) python=$(PYTHON)

install_env:
	@source $(CONDA_BASE)/etc/profile.d/conda.sh && \
	conda activate $(ENV) && \
	pip install light-the-torch && \
	ltt install torch torchvision \
	&& pip install -r $(REQUIREMENTS) \

install_detectron:
	@source $(CONDA_BASE)/etc/profile.d/conda.sh && \
	conda activate $(ENV) && \
	python -c "import torch, os ; \
 			   TORCH_VERSION = '.'.join(torch.__version__.split('.')[:2]);\
 			    CUDA_VERSION = torch.__version__.split('+')[-1]; \
 			    os.system( \
	f'pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/{CUDA_VERSION}/torch{TORCH_VERSION}/index.html')"

env: create_env install_env install_detectron

