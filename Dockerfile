# set base image
FROM continuumio/miniconda3:24.7.1-0

# set working directory
WORKDIR /app

# set environment variables
ENV PYTHONPATH=/app
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# install packages
COPY requirements.txt .
COPY environment.yaml .
RUN apt-get update \
    && apt install curl -y
RUN conda env create -f environment.yaml

# copy source code
COPY . .

# run script
CMD exec conda run --no-capture-output -n symmetry4CausalBounds python -u src/main.py