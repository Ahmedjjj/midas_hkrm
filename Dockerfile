FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6

COPY requirements.txt /requirements.txt
RUN pip install --upgrade --no-cache-dir  -r /requirements.txt

COPY MiDaS/ /midas
ENV PYTHONPATH=${PYTHONPATH}:/midas:/app



