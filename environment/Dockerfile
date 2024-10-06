# hash:sha256:1243fd4a68d689824668a06a212dd8a86ae7f6dd2c587d099b4a573c044f86c0
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
    ipywidgets==8.1.5 \
    jupyter==1.1.1 \
    matplotlib==3.9.2 \
    numpy==1.26.4 \
    pandas==2.2.3 \
    pathlib==1.0.1 \
    tensorboard==2.18.0 \
    torch==2.4.1 \
    tqdm==4.66.5

ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server
