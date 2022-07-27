# build the image required for setting up the repository
# Example run:
# $ docker build --rm -t wadhwaniai/pest-monitoring:v2 .
# Creates a docker image with desired dependencies

FROM pytorchlightning/pytorch_lightning:base-cuda-py3.8-torch1.8

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
    aufs-tools \
    automake \
    build-essential \
    curl \
    dpkg-sig \
    libcap-dev \
    libsqlite3-dev \
    mercurial \
    virtualenv \
    reprepro \
    tmux \
    vim \
    ruby1.9.1 && rm -rf /var/lib/apt/lists/*

# set the PYTHONPATH required for using the repository
ENV PYTHONPATH /workspace/pest-monitoring-new

# set actual working directory
WORKDIR /workspace/pest-monitoring-new

# copy the requirements file to the working directory
COPY requirements.txt .

# Install the required packages
RUN pip install --disable-pip-version-check -U -r requirements.txt
