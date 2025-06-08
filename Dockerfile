FROM nvcr.io/nvidia/pytorch:24.12-py3 as builder
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


# Build and install COLMAP.
RUN git clone https://github.com/colmap/colmap.git
RUN cd colmap && \
    git fetch https://github.com/colmap/colmap.git ${COLMAP_GIT_COMMIT} && \
    git checkout FETCH_HEAD && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        -DCMAKE_INSTALL_PREFIX=/colmap-install && \
    ninja install




FROM nvcr.io/nvidia/pytorch:24.12-py3 as runtime
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]


RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        libboost-program-options1.83.0 \
        libc6 \
        libceres4t64 \
        libfreeimage3 \
        libgcc-s1 \
        libgl1 \
        libglew2.2 \
        libgoogle-glog0v6t64 \
        libqt5core5a \
        libqt5gui5 \
        libqt5widgets5 \
        libcurl4

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
WORKDIR /hdd2/0321_block_drone_video/colmap
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Co-tracker 설치
COPY submodules/co-tracker ./submodules/co-tracker
RUN pip install -e ./submodules/co-tracker



# Install nvm, Node.js, and Claude Code
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash && \
    export NVM_DIR="$HOME/.nvm" && \
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" && \
    nvm install 22 && \
    nvm use 22 && \
    npm install -g @anthropic-ai/claude-code

# Add nvm to bashrc for interactive shells
RUN echo 'export NVM_DIR="$HOME/.nvm"' >> ~/.bashrc && \
    echo '[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"' >> ~/.bashrc

    

# 컨테이너 시작 시 bash 실행
COPY --from=builder /colmap-install/ /usr/local/

CMD ["/bin/bash"]   