ENVNAME=jellouli-env
REQUIREMENTS=requirements.txt
PYTHON=3.7

create_env:
	@conda create -n $(ENVNAME) python=$(PYTHON)

install_env:
	@conda init bash & conda activate $(ENVNAME) & pip install -qr $(REQUIREMENTS) 
