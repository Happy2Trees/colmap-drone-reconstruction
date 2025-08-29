FROM nvcr.io/nvidia/pytorch:24.12-py3
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]
####################################################################################################
# Prevent stop building ubuntu at time zone selection.
ARG COLMAP_GIT_COMMIT=3.11.1
ENV QT_XCB_GL_INTEGRATION=xcb_egl

# Set CUDA architectures for NVIDIA A6000 (Ampere, compute capability 8.6)
ARG CUDA_ARCHITECTURES=86

# Prepare and empty machine for building.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        git \
        cmake \
        ninja-build \
        build-essential \
        libboost-program-options-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libgmock-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libceres-dev

# install cudss
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
RUN  dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update
RUN apt-get -y install cudss


# install ceres-solver
WORKDIR /temp
RUN git clone --recurse-submodules -j8 https://github.com/ceres-solver/ceres-solver && \
    cd ceres-solver && \
    git checkout 46b4b3b002994ddb9d6fc72268c3e271243cd1df && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j 64 && \
    make install


# Build and install COLMAP. 3.11.1
RUN git clone https://github.com/colmap/colmap.git
RUN cd colmap && \
    git fetch https://github.com/colmap/colmap.git ${COLMAP_GIT_COMMIT} && \
    git checkout FETCH_HEAD && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
    ninja install




FROM nvcr.io/nvidia/pytorch:24.12-py3 as runtime
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]


RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        git \
        cmake \
        ninja-build \
        build-essential \
        libboost-program-options-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libgmock-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libceres-dev
WORKDIR /temp
RUN git clone --recursive https://github.com/colmap/glomap.git
RUN cd glomap && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} -DCMAKE_CXX_FLAGS='-mno-avx512f -mno-avx512dq -mno-avx512vl -Wno-error=array-bounds -Wno-error=stringop-overread' && \
    ninja && ninja install

######## PIP
# OS 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    screen \
    tzdata \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 타임존을 기본값으로 설정 (UTC 예시)
RUN echo "Etc/UTC" > /etc/timezone && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata


# 작업 디렉토리 설정
WORKDIR /temp
COPY requirements.txt ./
RUN pip install -r requirements.txt
# # # Co-tracker 설치
# # COPY submodules/co-tracker ./submodules/co-tracker
# # RUN pip install -e ./submodules/co-tracker


# # xformers
# RUN git clone --branch v0.0.29.post3 --recursive https://github.com/facebookresearch/xformers.git
# RUN git config --global --add safe.directory /temp/xformers && \
#     git config --global --add safe.directory /temp/xformers/third_party/flash-attention && \
#     git config --global --add safe.directory /temp/xformers/third_party/cutlass
# RUN cd xformers && echo '' > requirements.txt
# WORKDIR /temp/xformers
# RUN pip install ninja
# # cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS='-mno-avx512f -mno-avx512dq -mno-avx512vl -Wno-error=array-bounds -Wno-error=stringop-overread'
# # A6000 GPU (Ampere, compute capability 8.6)를 위한 설정
# ENV TORCH_CUDA_ARCH_LIST="8.6"
# ARG MAX_JOBS=16
# RUN pip install -vv -e .



CMD ["/bin/bash"]   