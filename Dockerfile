# set base image
FROM continuumio/miniconda3:24.7.1-0

# set working directory
WORKDIR /app

# set environment variables
ENV PYTHONPATH=/app
ENV CONDAENV=causal-bounds-da
ENV PYTHONVERSION=3.10.14
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# install packages
COPY requirements.txt .
RUN apt-get update \
    && apt install curl -y
# add channels, last added is with the highest priorety
RUN conda config --add channels pytorch
RUN conda config --add channels conda-forge
RUN conda config --add channels anaconda
# install with conda, install with pip on failure
RUN conda create --name "$CONDAENV" python="$PYTHONVERSION" --yes
RUN set -x && \
    conda init bash && \
    . ~/.bashrc && \
    conda activate ${CONDAENV} && \
    conda install pip --yes && \
    while read requirement; do conda install $requirement --yes \
    || pip install $requirement; done < requirements.txt

# copy source code
COPY . .

# run script
CMD exec conda run --no-capture-output -n $CONDAENV python -u src/main.py
