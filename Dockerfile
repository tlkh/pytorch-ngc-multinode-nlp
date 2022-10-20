FROM nvcr.io/nvidia/pytorch:22.09-py3
LABEL maintainer="Timothy Liu <timothy_liu@mymail.sutd.edu.sg>"
USER root
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -yq --no-install-recommends --no-upgrade \
    apt-utils \
    curl \
    wget \
    bzip2 \
    ca-certificates \
    build-essential \
    cmake \
    graphviz\
    git \
    nano \
    htop \
    zip \
    unzip \
    locales \
    libncurses5-dev \
    libncursesw5-dev \
    libopenblas-base \
    libopenblas-dev \
    libboost-all-dev \
    libsdl2-dev \
    swig \
    pkg-config \
    g++ \
    zlib1g-dev \
    patchelf \
    sudo \ 
    default-jdk \
    gcc g++-7 \
    openssh-client openssh-server \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers \
    libaio-dev && \
    apt-get purge -y hwloc-nox libhwloc-dev libhwloc-plugins \
    && apt-get autoremove -y \
    && apt-get clean && \
    rm -rf /tmp/* && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# Install Open MPI
RUN wget --progress=dot:mega -O /tmp/openmpi-4.1.4-bin.tar.gz https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.4.tar.gz && \
    cd /tmp && tar -zxf /tmp/openmpi-4.1.4-bin.tar.gz && \
    mkdir openmpi-4.1.4/build && cd openmpi-4.1.4/build && ../configure --prefix=/usr/local && \
    make -j all && make install && ldconfig && \
    mpirun --version

# Allow OpenSSH to talk to containers without asking for confirmation
RUN mkdir -p /var/run/sshd
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_WITH_PYTORCH=1 HOROVOD_GPU_OPERATIONS=NCCL \
    pip install horovod[pytorch] --upgrade --no-cache-dir && \
    horovodrun --check-build

RUN ldconfig && \
    pip install --no-cache-dir triton==1.0.0 && \
    DS_BUILD_OPS=0 \
    pip install --no-cache-dir \
    deepspeed --global-option="build_ext" --global-option="-j8"

RUN pip install --upgrade --no-cache-dir \
       tqdm ipywidgets jupyterlab wandb seaborn \
       && \
    jupyter lab clean

RUN pip install --upgrade --no-cache-dir \
       t2t-tuner==0.1.4 transformers diffusers accelerate datasets pytorch_lightning

RUN ldconfig

EXPOSE 8888

