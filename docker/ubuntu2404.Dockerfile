FROM ubuntu:noble-20240827.1 

# Update cache
RUN apt update && \
    apt install -y wget

# Install CUDA
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    apt install ./cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt update && \
    apt install -y cuda-toolkit

# Install dependencies for python
RUN apt update && \
    apt install -y build-essential \
    zlib1g-dev libncurses5-dev \
    libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev \
    libffi-dev

# Install Python
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/3.10.15/Python-3.10.15.tgz && \
    tar -xvf Python-3.10.15.tgz && \
    cd Python-3.10.15 && \
    ./configure --enable-optimizations --with-ensure-pip=install && \
    make -j 8 && \
    make install

# copy module
WORKDIR /app
COPY . /app

# install dependencies
RUN pip3 install -e .