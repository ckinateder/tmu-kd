#FROM ubuntu:22.04
FROM python:3.10.15-alpine3.20

# Optionally install other dependencies, tools, etc.
RUN apk add -qq git libffi-dev build-base 

WORKDIR /app
COPY . /app

# If you have a requirements.txt, install dependencies
RUN pip install -e .