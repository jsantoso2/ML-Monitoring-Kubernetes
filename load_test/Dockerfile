## Pull Python Image from DockerHub
FROM python:3.9-slim-buster

## Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONBUFFERED True

## Switch working directory
WORKDIR /load_test

## Copy requirements.txt to workdir
COPY requirements.txt ./requirements.txt

## Install Required Packages
RUN pip install -r requirements.txt

## Copy all files to directory
COPY . /load_test

## Expose Port
EXPOSE 8089

## Execute Locust load test
CMD ["locust", "-f", "locustfile.py", "--host", "http://127.0.0.1:3000"]
