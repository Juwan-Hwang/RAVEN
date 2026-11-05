FROM rust:1-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir "maturin>=1.4,<2.0"
WORKDIR /src
COPY . .
# Portable x86-64 (AVX2/FMA): the evaluation host (AMD EPYC 7F72, Zen 2) has no AVX-512.
ENV RUSTFLAGS="-C target-cpu=x86-64-v3"
RUN /opt/venv/bin/maturin build --release --features python -o /wheels

FROM python:3.11-slim-bookworm
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir numpy h5py /wheels/*.whl && rm -rf /wheels
WORKDIR /app
COPY sisap26/ /app/
CMD ["python", "/app/search.py", "--help"]
