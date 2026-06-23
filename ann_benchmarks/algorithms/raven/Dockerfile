# RAVEN ann-benchmarks Dockerfile
# 设计文档第六层模式一：ann_benchmarks/algorithms/yourlib/
#
# 构建步骤：
#   1. 基于 Rust 官方镜像编译 RAVEN 库 + benchmark 二进制
#   2. 基于 Python 镜像运行 ann-benchmarks 框架
#
# 用法（在 ann-benchmarks 仓库根目录）：
#   python3 run.py --algorithm raven --dataset fashion-mnist

FROM docker.m.daocloud.io/library/rust:1.85-slim AS builder

WORKDIR /raven
COPY . .

# 编译 release 二进制（含 AVX2 内核）
RUN cargo build --release --bin raven_ann_bench

# 提取编译产物
RUN cp target/release/raven_ann_bench /usr/local/bin/raven_ann_bench


FROM docker.m.daocloud.io/library/python:3.11-slim

# 安装 ann-benchmarks 依赖（不依赖 scikit-learn，用暴力 KNN）
# 使用国内 PyPI 镜像加速
RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    h5py \
    numpy

# 从 builder 阶段复制 RAVEN 二进制
COPY --from=builder /usr/local/bin/raven_ann_bench /usr/local/bin/raven_ann_bench

# 复制 Python wrapper
COPY ann_benchmarks/algorithms/raven/__init__.py /app/__init__.py

WORKDIR /app

# ann-benchmarks 框架会自动调用 RAVEN 类
ENV RAVEN_BIN=/usr/local/bin/raven_ann_bench
