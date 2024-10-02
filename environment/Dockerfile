# hash:sha256:ffd3c8710709298d9e76acfb647cd1de40a51bbdfbaf7df3875275811a20d02d
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch:2.1.0-cuda11.8-mambaforge23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    caveclient \
    cloudvolume \
    matplotlib \
    numpy \
    pytorch \
    torchvision \
    tqdm
