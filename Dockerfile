# set base image
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# set working directory
WORKDIR /app

# set environment variables
ENV PYTHONPATH=/app
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# install locked dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# copy source code
COPY . .

# run script
CMD ["uv", "run", "--frozen", "python", "-u", "src/main.py"]
