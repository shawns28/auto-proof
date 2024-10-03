# hash:sha256:4891d041510d3c2a71036bf288edbaeae54fe071c7f34903a280dfdb167d1361
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/cuda-miniconda-jupyterlab:latest

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda install -y \
        python==3.10.9 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    caveclient==5.30.1 \
    cloud-volume==10.5.0 \
    matplotlib==3.9.2 \
    numpy==1.26.4 \
    pandas==2.2.3 \
    pathlib==1.0.1 \
    tensorboard \
    torch==2.4.1
