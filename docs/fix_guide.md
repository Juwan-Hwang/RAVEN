# RAVEN 工程问题修复指南

> **原则**：每个修复必须有 before/after 实验数据，确认优化有效且无回归。
> 修复按优先级排序，逐项提交，方便回退。

---

## 实验基准线

所有修复统一使用 SIFT1M 数据集和 `quick_recall_check` 验证：

```bash
# 基准线实验（修复前记录一次）
cargo build --release --bin quick_recall_check 2>&1 | tail -3
./target/release/quick_recall_check
```

**基准线参数**（来自最近一次实验）：
- recall@10 = 0.9517
- QPS = 2562
- 建图时间 = 852s

**验收标准**：
- recall 不低于 0.95（允许 ±0.005）
- QPS 不低于 2500（允许 ±5%）
- 单元测试全部通过

---

## FIX-1: 默认配置 avq+L2 冲突

**问题**：`config.rs:103` 默认 `distance="l2"` + `avq=true`，但 `rules.rs:137` 规则判定互斥，导致默认配置无法通过验证。

**影响**：任何使用默认配置的入口（包括测试）会触发规则冲突，行为不可预期。

**修复方案**：将默认 `avq` 改为 `false`，或将默认 `distance` 改为 `"ip"`。

```rust
// config.rs:115 — 改 avq 默认值为 false
avq: false,  // 默认关闭 AVQ，显式开启时需匹配 ip 距离
```

**验证步骤**：
1. `cargo test --lib` — 全部通过
2. `cargo clippy --lib -- -D warnings` — 无警告
3. `quick_recall_check` — recall/QPS 无变化（AVQ 在当前实验中未启用）

**预期风险**：🟢 极低。当前所有实验均手动指定参数，不受默认值影响。

---

## FIX-2: ann-benchmarks wrapper 索引复用

**问题**：`__init__.py` 的 `query()` 每次调用都传 `--train` 但不传 `--save`/`--load`，导致每次查询都从头建图。

**影响**：ann-benchmarks 评测时 QPS 被重复建图时间严重拉低，无法获得真实查询性能。

**修复方案**：
1. `fit()` 结束时传 `--save <path>` 保存索引
2. `query()`/`query_batch()` 传 `--load <path>` 跳过建图

```python
# __init__.py — fit() 保存索引
def fit(self, X):
    ...
    self._index_file = tempfile.mktemp(suffix=".idx", prefix="raven_idx_")
    cmd = [..., "--save", self._index_file]
    ...

# __init__.py — query() 复用索引
def query(self, q, k):
    cmd = [self.raven_bin, "--load", self._index_file, ...]
    ...
```

**验证步骤**：
1. 单元测试：`cargo test --lib`
2. 手动验证：Python 中调用 `fit()` → 多次 `query()`，确认第二次查询明显更快
3. `raven_ann_bench.rs` 需确认 `--save`/`--load` 参数已正确传递到 `VamanaGraph::save()`/`VamanaGraph::load()`

**预期风险**：🟡 中等。需确认序列化/反序列化后 recall 不变（已有 `serialize_roundtrip` 测试覆盖）。

---

## FIX-3: 距离分发死代码清理

**问题**：`dispatch.rs:60-66` 的 `match dim` 所有分支都返回 `l2_simd`，高频维度分支是死代码。

**影响**：无功能影响，但误导读者认为有维度特化。

**修复方案**：简化为单行，移除无意义的 match。

```rust
pub fn select_l2(dim: usize) -> fn(&[f32], &[f32]) -> f32 {
    let _ = dim; // 当前所有维度统一走 l2_simd
    l2_simd
}
```

**验证步骤**：
1. `cargo test --lib` — 通过
2. `cargo clippy --lib -- -D warnings` — 无警告
3. `quick_recall_check` — recall/QPS 无变化

**预期风险**：🟢 极低。纯代码等价重构。

---

## FIX-4: 未使用变量 / 警告清理

**问题**：`avx2.rs:183`（`expected`）、`opq.rs:327,339`（`eigenvectors`）3 个 unused variable 警告。

**影响**：CI clippy 步骤会失败。

**修复方案**：前缀 `_` 或移除未使用的绑定。

```rust
// avx2.rs:183
let _expected = 27.0f32;

// opq.rs:327,339
let (eigenvalues, _eigenvectors) = jacobi_eigen(...);
```

**验证步骤**：
1. `cargo clippy --lib -- -D warnings` — 通过
2. `cargo test --lib` — 通过

**预期风险**：🟢 极低。纯警告修复。

---

## FIX-5: RP-Tuning SchemeB/C 明确标注未实现

**问题**：`rp_tuning.rs:118` SchemeB/C 静默回退到 SchemeA，用户可能不知道。

**影响**：文档/接口暗示三种方案可用，实际只有 A。

**修复方案**：
1. 在 `RPTuningStorageScheme` 的 `SchemeB`/`SchemeC` 变体上加 `#[deprecated]` 或 doc 注释标注未实现
2. 回退日志保持（已有 `eprintln!`）

```rust
/// 方案 B：需构建期保留候选集（当前未实现，回退到 SchemeA）
SchemeB,
/// 方案 C：需构建期保留候选集（当前未实现，回退到 SchemeA）
SchemeC,
```

**验证步骤**：
1. `cargo test --lib` — 通过
2. `cargo doc` — 文档生成无警告

**预期风险**：🟢 极低。纯文档标注。

---

## FIX-6: big-ann/SSD 预留接口标注

**问题**：`Cargo.toml` 中 `big_ann = []` feature flag 无对应实现。

**影响**：README 如果提及 big-ann 会造成误解。

**修复方案**：
1. README 中明确标注 big-ann 为"计划中，未实现"
2. Cargo.toml feature 注释保持现状（预留本身无害）

**验证步骤**：
1. 检查 README 无误导性描述

**预期风险**：🟢 极低。纯文档修正。

---

## FIX-7: README 诚实标注项目状态

**问题**：README 性能数据和功能描述可能暗示已完整实现。

**修复方案**：在 README 添加"项目状态"小节，明确标注：
- 核心库（L1-L3）：可用于研究和实验
- ann-benchmarks wrapper：基础可用，有已知性能问题
- big-ann/SSD：计划中，未实现
- RP-Tuning SchemeB/C：未实现，回退到 SchemeA
- GEMM 批量路径：标量回退，未接入真正 GEMM

**验证步骤**：
1. 阅读 README 确认无误导

**预期风险**：🟢 极低。纯文档。

---

## 执行顺序

| 序号 | 修复项 | 需要实验 | 预期风险 | 依赖 |
|------|--------|----------|----------|------|
| FIX-4 | 警告清理 | 否 | 🟢 | 无 |
| FIX-3 | 距离分发死代码 | 是（quick_recall_check） | 🟢 | 无 |
| FIX-1 | 默认配置冲突 | 是（quick_recall_check） | 🟢 | 无 |
| FIX-5 | SchemeB/C 标注 | 否 | 🟢 | 无 |
| FIX-6 | big-ann 标注 | 否 | 🟢 | 无 |
| FIX-7 | README 诚实标注 | 否 | 🟢 | FIX-5, FIX-6 |
| FIX-2 | wrapper 索引复用 | 是（wrapper smoke test） | 🟡 | 无 |

**建议**：FIX-4 → FIX-3 → FIX-1 → FIX-5 → FIX-6 → FIX-7 → FIX-2，逐项提交。
