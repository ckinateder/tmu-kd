FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS cuda

ENV DEBIAN_FRONTEND=noninteractive

# Add cuda to path
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV CUDA_ROOT="/usr/local/cuda"
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64

# Build python from source
FROM cuda AS python

RUN apt update && \
    apt install -y wget git

# Install dependencies needed to build python
RUN apt install -y build-essential \
    zlib1g-dev libncurses5-dev \
    libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev \
    libffi-dev libgl1-mesa-glx libgl1 libglib2.0-0

# Get thread arg
ARG THREADS=8
ARG PYTHON_VERSION=3.11.10

# Install Python
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xvf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations --with-ensure-pip=install && \
    make -j ${THREADS} && \
    make install

# Link python3 to python
RUN ln -s /usr/local/bin/python3 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3 /usr/local/bin/pip

# Install local python packages
FROM python AS local-build

# Copy the local package files to the container's workspace
WORKDIR /app
COPY . /app

# install dependencies
RUN pip install -e .
RUN pip install -r cuda-requirements.txt
RUN python -m pip install pycuda opencv-python tensorflow[and-cuda] torch torch-tensorrt tensorrt --extra-index-url https://download.pytorch.org/whl/cu124
