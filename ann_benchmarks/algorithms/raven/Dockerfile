FROM ann-benchmarks

# 安装 Rust 工具链依赖（base 镜像已有 git/build-essential/python3-pip）
# python3-dev: PyO3 编译需要 Python.h
# pkg-config: maturin 探测 Python 需要它
RUN apt-get update && apt-get install -y curl pkg-config python3-dev

# 安装 Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# 安装 maturin（PyO3 构建工具）
RUN pip3 install maturin numpy

# 从 GitHub clone RAVEN 源码
# 注意：ann-benchmarks 的构建上下文是 ann-benchmarks 仓库根目录，
# 不能用 COPY（会复制整个 ann-benchmarks），参考 diskann 的 git clone 方式
RUN git clone --depth 1 https://github.com/Juwan-Hwang/RAVEN.git /raven

# 覆盖本地修改的 python.rs（use_adaptive flag 优化：查询时切换 adaptive/固定ef，避免重复建图）
COPY ann_benchmarks/algorithms/raven/python.rs /raven/src/python.rs

WORKDIR /raven

# 编译 RAVEN Python 扩展（release + native CPU target）
# target-cpu=native 让 Docker 内编译用 host CPU 特性（AVX-512 等）
RUN RUSTFLAGS="-C target-cpu=native" maturin build --release --features python

# 安装编译好的 wheel
RUN pip3 install target/wheels/*.whl

# 验证导入
RUN python3 -c 'import raven; print("RAVEN Python binding loaded")'

# 回到 base 镜像的 WORKDIR，run_algorithm.py 在 /home/app
WORKDIR /home/app
