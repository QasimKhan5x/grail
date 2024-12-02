# Base image
FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install tmux

# Add the deadsnakes PPA for Python 3.10
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    wget \
    git \
    zip \
    unzip \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Update alternatives to set Python 3.10 as the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --config python3 && \
    python3 --version

# Install the latest version of pip manually
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# Copy the entire current folder to the container
COPY . /home/

# Set working directory
WORKDIR /home/grail

# Install Python dependencies
RUN pip install -r requirements.txt

# Install additional utilities commonly used in deep learning projects
RUN pip install jupyterlab ipython matplotlib seaborn

# Expose a default port for Jupyter
EXPOSE 8888

# Default command
CMD ["bash"]

# Run this --> docker run --gpus all -it -p 8899:8888 -v $(pwd):/home/grail grail
