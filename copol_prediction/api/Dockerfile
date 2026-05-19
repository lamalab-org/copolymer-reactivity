# Dockerfile for Copolymerization Prediction API
# Multi-stage build for smaller final image with XTB support via xtb-python

# ============================================================================
# Stage 1: Builder - Install dependencies and build the package
# ============================================================================
# Pin to a specific tag rather than `:latest` so rebuilds are reproducible
# across environments. Bump deliberately; do not chase :latest. (See
# https://pythonspeed.com/articles/dockerizing-python-is-hard/.)
FROM continuumio/miniconda3:26.3.2-2 as builder

# Set working directory
WORKDIR /build

# Install system dependencies required for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment with Python 3.11 and xtb-python
RUN conda create -n copol python=3.11 -y && \
    conda install -n copol -c conda-forge xtb-python -y && \
    conda clean -afy

# Activate environment and install Python dependencies
SHELL ["conda", "run", "-n", "copol", "/bin/bash", "-c"]

# Install Python deps BEFORE copying source so source-only changes don't bust
# the (slow, ~all the dependencies) pip install layer cache.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
COPY copol_prediction/api/requirements.txt /build/api-requirements.txt
RUN pip install --no-cache-dir -r /build/api-requirements.txt

# Now copy source and install the local package. --no-deps because every
# dependency this package declares is already in api-requirements.txt
# (verified for current pyproject.toml). This step is cheap, so re-running
# it on every source change is fine.
COPY pyproject.toml LICENSE /build/
COPY src /build/src
RUN pip install --no-cache-dir --no-deps -e .

# ============================================================================
# Stage 2: Runtime - Create minimal runtime image
# ============================================================================
FROM continuumio/miniconda3:26.3.2-2

# Build provenance — surfaced at runtime via /health. CI passes real values
# via --build-arg in .github/workflows/docker-image.yml; local docker builds
# without --build-arg get "unknown", which is what /health reports too.
ARG GIT_SHA=unknown
ARG GIT_BRANCH=unknown
ARG BUILD_TIME=unknown

# Metadata
LABEL maintainer="Copolymerization Prediction API"
LABEL description="FastAPI REST API for copolymerization reactivity prediction with XTB support"
LABEL version="1.0.0"
# Standard OCI labels so `docker inspect` / GHCR UI / image-vuln scanners
# can see the same provenance without booting the container.
LABEL org.opencontainers.image.revision="${GIT_SHA}"
LABEL org.opencontainers.image.ref.name="${GIT_BRANCH}"
LABEL org.opencontainers.image.created="${BUILD_TIME}"
LABEL org.opencontainers.image.source="https://github.com/lamalab-org/copolymer-reactivity"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/artifacts/model_bundle \
    PORT=8000 \
    OMP_NUM_THREADS=2 \
    OMP_STACKSIZE=1G \
    OMP_MAX_ACTIVE_LEVELS=1 \
    PATH=/opt/conda/envs/copol/bin:$PATH \
    GIT_SHA=${GIT_SHA} \
    GIT_BRANCH=${GIT_BRANCH} \
    BUILD_TIME=${BUILD_TIME}

# Install runtime system dependencies. `patchelf` is used below to clear the
# executable-stack flag on the xtb shared libraries, which is rejected by
# modern Linux kernels (ImportError: cannot enable executable stack).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Copy conda environment from builder
COPY --from=builder /opt/conda/envs/copol /opt/conda/envs/copol

# Clear the executable-stack flag on xtb's native libs.
RUN find /opt/conda/envs/copol/lib \
        \( -name 'libxtb*.so*' -o -name 'libmctc*.so*' \
           -o -name 'libmstore*.so*' -o -name 'libcontainer*.so*' \) \
        -exec patchelf --clear-execstack {} \;

# Create non-root user for security
RUN useradd -m -u 1000 apiuser && \
    mkdir -p /app && \
    chown -R apiuser:apiuser /app

# Set working directory
WORKDIR /app

# Copy application code and config
COPY --chown=apiuser:apiuser copol_prediction/api/app.py /app/
COPY --chown=apiuser:apiuser copol_prediction/api/morfeus_patch.py /app/
COPY --chown=apiuser:apiuser copol_prediction/api/baseline_lookup.py /app/
COPY --chown=apiuser:apiuser copol_prediction/api/reaction_optimization.py /app/

# Copy data files
COPY --chown=apiuser:apiuser copol_prediction/api/data /app/data

# Copy config files  
COPY --chown=apiuser:apiuser copol_prediction/api/config /app/config

# Copy molecule properties cache (optional, for faster monomer lookups)
COPY --chown=apiuser:apiuser copol_prediction/api/molecule_properties /app/molecule_properties

# Copy model artifacts
COPY --chown=apiuser:apiuser copol_prediction/artifacts /app/artifacts

# Copy dataset (used for DOI checking)
COPY --chown=apiuser:apiuser copol_prediction/processed_data.csv /app/processed_data.csv

# Copy augmented negative data points used by the baseline-lookup pool
COPY --chown=apiuser:apiuser copol_prediction/filter/artificial_datapoints/processed_combined_augmented.csv /app/artificial_datapoints.csv

# Copy source packages (needed for imports)
COPY --chown=apiuser:apiuser src /app/src

# Install the local package in the runtime stage. --no-deps because every
# runtime dep is already in the conda env copied from the builder stage;
# this step just registers the editable `copolpredictor` import path.
COPY --chown=apiuser:apiuser pyproject.toml /app/
SHELL ["conda", "run", "-n", "copol", "/bin/bash", "-c"]
RUN pip install --no-cache-dir --no-deps -e .

# Create cache directory
RUN mkdir -p /app/cache && chown apiuser:apiuser /app/cache

# Switch to non-root user
USER apiuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD conda run -n copol python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Start the API
# Single worker recommended for XTB calculations to avoid resource contention
# Use conda run to ensure environment is activated
CMD ["conda", "run", "--no-capture-output", "-n", "copol", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "300"]

