# Opt-in CPU-execution image. Does not download models or training data.
# PyTorch dependencies use the explicit CPU-wheel source in the lock.
FROM python:3.11-slim AS builder
WORKDIR /app
RUN python -m pip install --no-cache-dir uv==0.13.8
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --extra ml --no-dev --no-editable

FROM python:3.11-slim AS runtime
ENV PATH="/app/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=-1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1
RUN useradd --create-home --uid 10001 app
COPY --from=builder /app/.venv /app/.venv
COPY scripts/demo.py /app/demo.py
COPY scripts /app/scripts
USER app
WORKDIR /home/app
ENTRYPOINT ["python", "/app/demo.py"]
CMD ["/tmp/synthetic-run"]
