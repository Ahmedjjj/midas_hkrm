SHELL := /bin/bash

ENV=jellouli-env
REQUIREMENTS=requirements.txt
PYTHON=3.7
DOCKERIMAGE=jellouli_docker
CONDA_BASE=$(shell conda info --base)

create_env:
	@conda create -n $(ENV) python=$(PYTHON)

install_env:
	source $(CONDA_BASE)/etc/profile.d/conda.sh && \
	conda activate $(ENV) && pip install -r $(REQUIREMENTS)

image:
	@docker build -t $(DOCKERIMAGE) .

notebook: image
	@docker run --rm --network host -v $(shell pwd):/app -w /app \
	$(DOCKERIMAGE) jupyter notebook --allow-root --no-browser

