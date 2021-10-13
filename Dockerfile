FROM python:3.7

COPY requirements.txt /requirements.txt
RUN pip install --upgrade --no-cache-dir  -r /requirements.txt

RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6

