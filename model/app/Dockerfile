## Pull Python Image from DockerHub
FROM python:3.9-slim-buster

## Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONBUFFERED True

## Switch working directory
WORKDIR /app

## Copy requirements.txt to workdir
COPY requirements.txt ./requirements.txt

## Install Required Packages
RUN pip install -r requirements.txt

## Copy all files to directory
COPY . /app

## Expose Port
EXPOSE 8080

## Set Env Variable
ENV PROMETHEUS_MULTIPROC_DIR=/tmp

## Execute with gunicorn server (config is required for Prometheus Exporter)
CMD exec gunicorn --config config.py --bind 0.0.0.0:8080 --workers 1 --threads 4 flask_app:app


## Deploy Locally
#CMD python flask_app.py 