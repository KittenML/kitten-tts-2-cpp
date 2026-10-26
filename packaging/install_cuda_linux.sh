#!/bin/sh
# Install the CUDA 12.8 compiler, runtime, cuBLAS and NVRTC development files in a manylinux_2_34 (RHEL 9 family)
# build container. LibTorch's CMake package links NVRTC when torch is a CUDA build, so configuring fails without it.
set -eu
# EPEL is not needed and its mirrors can fail mid-sync, so skip it
dnf install -y --disablerepo='epel*' dnf-plugins-core
dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
dnf install -y --disablerepo='epel*' cuda-nvcc-12-8 cuda-cudart-devel-12-8 libcublas-devel-12-8 cuda-nvrtc-devel-12-8 cuda-cccl-12-8
/usr/local/cuda-12.8/bin/nvcc --version
