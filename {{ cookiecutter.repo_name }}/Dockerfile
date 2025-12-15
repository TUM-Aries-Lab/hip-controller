# syntax=docker/dockerfile:1.7-labs
FROM python:3.12-slim

# Fast, deterministic behavior
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Select the package and version at build time
ARG PKG={{ cookiecutter.module_name }}
ARG VER=latest
ENV PKG=${PKG} VER=${VER}

# Install UV
RUN --mount=type=cache,target=/root/.cache \
    pip install uv

# Install Python dependencies using UV instead of pip
# UV uses its own resolver + is ~10x faster + fully deterministic.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$VER" = "latest" ]; then \
        uv pip install --system "$PKG"; \
    else \
        uv pip install --system "$PKG==$VER"; \
    fi

# Simple smoke test script
WORKDIR /app
RUN printf '%s\n' \
  "import importlib, os, sys" \
  "m = importlib.import_module(os.environ.get('PKG', '${PKG}'))" \
  "print('✅ import ok:', getattr(m, '__version__', 'unknown'), 'on', sys.version)" \
  > smoke.py

CMD ["uv", "run", "python", "smoke.py"]
