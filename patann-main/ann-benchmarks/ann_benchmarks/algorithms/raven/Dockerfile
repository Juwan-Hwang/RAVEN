FROM ann-benchmarks

# 安装 Rust 工具链
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# 安装 maturin（PyO3 构建工具）
RUN pip3 install maturin numpy

# 复制 RAVEN 源码
COPY . /raven
WORKDIR /raven

# 编译 RAVEN Python 扩展（release 优化 + AVX2 target）
# ann-benchmarks Docker 环境通常是 x86_64，支持 AVX2
RUN RUSTFLAGS="-C target-cpu=native" maturin build --release --features python

# 安装编译好的 wheel
RUN pip3 install target/wheels/*.whl

# 验证导入
RUN python3 -c 'import raven; print("RAVEN Python binding loaded")'

WORKDIR /app
