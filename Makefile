ENV=jellouli-env
REQUIREMENTS=requirements.txt
PYTHON=3.7
DOCKERIMAGE=jellouli_docker

create_env:
	@conda create -n $(ENV) python=$(PYTHON)

install_env:
	@conda init bash & conda activate $(ENV) & pip install -qr $(REQUIREMENTS)

image:
	@docker build -t $(DOCKERIMAGE) .

notebook:
	@docker run --rm --network host -v $(shell pwd):/app -w /app \
	$(DOCKERIMAGE) jupyter notebook --allow-root --no-browser