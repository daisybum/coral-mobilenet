# ─── 베이스 이미지를 devel 버전으로 변경 ───────────────────
ARG BASE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${BASE}

# ─── 비대화식 모드 설정 ────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive

# ─── 시스템 패키지 설치 (Python3, pip 포함) ────────────────
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip build-essential git vim wget gnupg curl ca-certificates locales && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# ─── CUDA 레포지토리 등록 & CUDA 툴킷 + cuDNN 설치 ───────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl wget && \
    # 1) NVIDIA의 encapsulated keyring 파일을 받아서 keyrings에 배치
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-archive-keyring.gpg \
      -O /usr/share/keyrings/cuda-archive-keyring.gpg && \
    # 2) apt 소스리스트에 signed-by 옵션으로 CUDA 레포지토리 추가
    echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \
      > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && \
    # 3) CUDA 툴킷 및 cuDNN 런타임+헤더 한 번에 설치
    apt-get install -y --no-install-recommends \
        cuda-toolkit-12-4 \
        libcudnn8 libcudnn8-dev && \
    rm -rf /var/lib/apt/lists/*

# ─── 로케일 설정 (UTF-8) ──────────────────────────────────
RUN sed -i 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# ─── 파이썬 패키지 설치 ────────────────────────────────────
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ─── 작업 디렉토리 및 소스 복사 ─────────────────────────────
WORKDIR /workspace
COPY . .

# ─── 기본 실행 명령 ───────────────────────────────────────
CMD ["tail", "-f", "/dev/null"]
