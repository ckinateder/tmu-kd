FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS cuda

# Add cuda to path
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV CUDA_ROOT="/usr/local/cuda"

# Build python from source
FROM cuda AS python

RUN apt update && \
    apt install -y wget git

# Install dependencies needed to build python
RUN apt install -y build-essential \
    zlib1g-dev libncurses5-dev \
    libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev \
    libffi-dev

# Get thread arg
ARG THREADS=8

# Install Python
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/3.10.15/Python-3.10.15.tgz && \
    tar -xvf Python-3.10.15.tgz && \
    cd Python-3.10.15 && \
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

# Remove App files, we don't need them anymore
RUN rm -rf /app

# Set the working directory
WORKDIR /app