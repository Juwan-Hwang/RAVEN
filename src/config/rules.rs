//! 规则驱动 Auto-tuner（含 DAG 冲突校验）
//!
//! 设计文档第六层：
//! 规则层定位：合法性 / 常识性剪枝（skip），不表达 preference
//! 所有配置入口必须先合并成唯一最终配置，再统一校验，校验后不允许任何路径静默覆盖
//!
//! rules! {
//!     skip if avq == true && distance == L2;
//!     skip if avx512 == true && dim % 16 != 0;
//!     skip if gemm_path == true && candidate_count <= gemm_threshold;
//!     skip if batch_mode == true && !cfg!(feature = "batch_mode");
//! }
//!
//! Week 7-8 扩展规则（打榜前参数校验）：
//!     error if alpha <= 0;
//!     warn  if alpha >= 1.0;  // 性能可能下降
//!     error if codebook_k not power of 2;
//!     error if sub_dim not divides dim;
//!     error if top_n < k;
//!     error if avq_alpha not in [0.0, 1.0];

use thiserror::Error;
use super::Config;

/// 规则校验错误
#[derive(Debug, Error)]
#[allow(missing_docs)]
pub enum ConflictError {
    #[error("rule '{rule}' violated: {reason}")]
    RuleViolated { rule: String, reason: String },
    #[error("config load error: {0}")]
    ConfigLoad(String),
}

/// 规则严重级别
///
/// 设计文档：规则层只做 skip（合法性），不表达 preference
/// Error：违反规则，阻止执行
/// Warning：潜在问题，输出警告但不阻止执行
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleSeverity {
    /// 错误：违反规则，阻止执行
    Error,
    /// 警告：潜在问题，不阻止执行
    Warning,
}

/// 规则定义
///
/// 设计文档：规则层只做 skip，不做 prefer
#[derive(Debug, Clone)]
pub struct Rule {
    /// 规则名称
    pub name: String,
    /// 规则描述
    pub description: String,
    /// 校验函数：返回 true 表示通过，false 表示违反
    pub check: fn(&Config) -> bool,
    /// 违反时的原因
    pub violation_reason: String,
    /// 严重级别
    pub severity: RuleSeverity,
    /// 依赖的规则名列表（DAG 校验用）
    /// 设计文档：含 DAG 冲突校验
    pub depends_on: Vec<String>,
}

impl Rule {
    /// 创建 Error 级别规则
    pub fn new(
        name: &str,
        description: &str,
        check: fn(&Config) -> bool,
        violation_reason: &str,
    ) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            check,
            violation_reason: violation_reason.to_string(),
            severity: RuleSeverity::Error,
            depends_on: Vec::new(),
        }
    }

    /// 创建 Warning 级别规则
    pub fn new_warning(
        name: &str,
        description: &str,
        check: fn(&Config) -> bool,
        violation_reason: &str,
    ) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            check,
            violation_reason: violation_reason.to_string(),
            severity: RuleSeverity::Warning,
            depends_on: Vec::new(),
        }
    }

    /// 设置依赖规则（DAG 校验用）
    /// 设计文档：含 DAG 冲突校验
    pub fn with_depends_on(mut self, deps: &[&str]) -> Self {
        self.depends_on = deps.iter().map(|s| s.to_string()).collect();
        self
    }
}

/// 规则引擎
///
/// 设计文档第六层：规则驱动 Auto-tuner（含 DAG 冲突校验）
pub struct RuleEngine {
    rules: Vec<Rule>,
}

impl Default for RuleEngine {
    fn default() -> Self {
        Self::with_standard_rules()
    }
}

/// 判断 n 是否是 2 的幂次
fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

impl RuleEngine {
    /// 创建带标准规则的引擎
    ///
    /// 设计文档 rules! 块的四条规则 + Week 7-8 扩展规则
    pub fn with_standard_rules() -> Self {
        let rules = vec![
            // === 设计文档原始规则 ===
            // 设计文档：skip if avq == true && distance == L2
            Rule::new(
                "avq_l2_conflict",
                "AVQ 与 L2 距离互斥",
                |cfg| !(cfg.avq && cfg.distance == "l2"),
                "AVQ 模式与 L2 距离互斥（设计文档 F.8）",
            ),
            // 设计文档：skip if avx512 == true && dim % 16 != 0
            Rule::new(
                "avx512_dim_alignment",
                "AVX-512 要求维度是 16 的倍数",
                |cfg| !cfg.avx512 || cfg.dim % 16 == 0,
                "AVX-512 要求维度是 16 的倍数（__m512 = 16 × f32）",
            ),
            // 设计文档：skip if gemm_path == true && candidate_count <= gemm_threshold
            Rule::new(
                "gemm_threshold_check",
                "GEMM 路径要求候选数超过阈值",
                |cfg| !cfg.gemm_path || cfg.candidate_count > cfg.gemm_threshold,
                "候选数未超过 GEMM 阈值，不应走 GEMM 路径",
            ),
            // 设计文档：skip if batch_mode == true && !cfg!(feature = "batch_mode")
            Rule::new(
                "batch_mode_feature_gate",
                "批量模式需要 batch_mode feature",
                |cfg| !cfg.batch_mode || cfg!(feature = "batch_mode"),
                "批量模式需要启用 batch_mode feature flag",
            ),
            // === Week 7-8 扩展规则（打榜前参数校验）===
            // 规则 1a：α > 0（Error）
            Rule::new(
                "alpha_positive",
                "α 参数必须大于 0",
                |cfg| cfg.alpha > 0.0,
                "α ≤ 0 非法：剪枝参数必须为正数",
            ),
            // 规则 1b：α ≥ 1.0 警告性能可能下降（Warning）
            Rule::new_warning(
                "alpha_high_warning",
                "α ≥ 1.0 时性能可能下降",
                |cfg| cfg.alpha < 1.0,
                "α ≥ 1.0：剪枝更宽松，图度数增大，QPS 可能下降（Week 7 实测：α=1.2 QPS 比 α=1.0 低 5-15%）",
            ),
            // 规则 2：K（codebook 大小）必须是 2 的幂次
            Rule::new(
                "codebook_k_power_of_two",
                "AVQ codebook K 必须是 2 的幂次",
                |cfg| !cfg.avq || is_power_of_two(cfg.codebook_k),
                "codebook K 必须是 2 的幂次（PQ 编码要求：K=256 → 8-bit 编码）",
            ),
            // 规则 3：sub_dim 必须整除 dim
            Rule::new(
                "sub_dim_divides_dim",
                "sub_dim 必须整除 dim",
                |cfg| cfg.sub_dim > 0 && cfg.dim % cfg.sub_dim == 0,
                "sub_dim 必须整除 dim（AVQ 子空间划分要求）",
            ),
            // 规则 4：top_n ≥ k
            Rule::new(
                "top_n_ge_k",
                "rerank 候选数 top_n 必须 ≥ 最终返回数 k",
                |cfg| cfg.top_n >= cfg.k,
                "top_n < k：rerank 候选数不足以填充最终结果",
            ),
            // 规则 5：AVQ α ∈ [0.0, 1.0]
            Rule::new(
                "avq_alpha_range",
                "AVQ 混合权重 α 必须在 [0.0, 1.0] 范围内",
                |cfg| !cfg.avq || (cfg.avq_alpha >= 0.0 && cfg.avq_alpha <= 1.0),
                "AVQ α 不在 [0.0, 1.0]：混合损失权重无效（α * recon + (1-α) * ret）",
            ),
        ];
        Self { rules }
    }

    /// 添加自定义规则
    pub fn add_rule(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    /// 创建空引擎（用于自定义规则集）
    pub fn new_empty() -> Self {
        Self { rules: Vec::new() }
    }

    /// DAG 冲突校验（设计文档：含 DAG 冲突校验）
    ///
    /// 检测规则依赖关系是否存在环（循环依赖）
    /// 如果存在环，返回错误
    pub fn validate_dag(&self) -> Result<(), ConflictError> {
        use std::collections::{HashMap, VecDeque};

        // 构建规则名到索引的映射
        let name_to_idx: HashMap<&str, usize> = self.rules
            .iter()
            .enumerate()
            .map(|(i, r)| (r.name.as_str(), i))
            .collect();

        // 构建邻接表
        let n = self.rules.len();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, rule) in self.rules.iter().enumerate() {
            for dep in &rule.depends_on {
                if let Some(&dep_idx) = name_to_idx.get(dep.as_str()) {
                    adj[dep_idx].push(i);
                }
            }
        }

        // 拓扑排序检测环（Kahn 算法）
        let mut in_degree = vec![0usize; n];
        for edges in &adj {
            for &v in edges {
                in_degree[v] += 1;
            }
        }
        let mut queue: VecDeque<usize> = VecDeque::new();
        for i in 0..n {
            if in_degree[i] == 0 {
                queue.push_back(i);
            }
        }
        let mut visited = 0;
        while let Some(u) = queue.pop_front() {
            visited += 1;
            for &v in &adj[u] {
                in_degree[v] -= 1;
                if in_degree[v] == 0 {
                    queue.push_back(v);
                }
            }
        }
        if visited != n {
            return Err(ConflictError::RuleViolated {
                rule: "dag_cycle".to_string(),
                reason: "规则依赖关系存在环（循环依赖）".to_string(),
            });
        }
        Ok(())
    }

    /// 校验 Error 规则
    ///
    /// 设计文档：合并完成后统一校验一次
    /// 只检查 Error 级别规则，Warning 级别用 check_warnings()
    pub fn validate(&self, cfg: &Config) -> Result<(), ConflictError> {
        // 设计文档：含 DAG 冲突校验
        self.validate_dag()?;
        for rule in &self.rules {
            if rule.severity == RuleSeverity::Error && !(rule.check)(cfg) {
                return Err(ConflictError::RuleViolated {
                    rule: rule.name.clone(),
                    reason: rule.violation_reason.clone(),
                });
            }
        }
        Ok(())
    }

    /// 检查 Warning 规则，返回违反的警告列表
    ///
    /// 设计文档：规则层只做 skip（合法性），Warning 不阻止执行
    pub fn check_warnings(&self, cfg: &Config) -> Vec<&Rule> {
        self.rules
            .iter()
            .filter(|r| r.severity == RuleSeverity::Warning && !(r.check)(cfg))
            .collect()
    }

    /// 校验并输出警告（tracing::warn）
    ///
    /// 便捷方法：validate + check_warnings + tracing::warn
    pub fn validate_and_warn(&self, cfg: &Config) -> Result<(), ConflictError> {
        self.validate(cfg)?;
        for warning in self.check_warnings(cfg) {
            tracing::warn!(
                rule = %warning.name,
                reason = %warning.violation_reason,
                "configuration warning"
            );
        }
        Ok(())
    }

    /// 规则数
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_engine_has_standard_rules() {
        let engine = RuleEngine::default();
        // 4 条原始 + 6 条扩展 = 10 条
        assert_eq!(engine.len(), 10);
    }

    #[test]
    fn validate_default_config() {
        let engine = RuleEngine::default();
        let cfg = Config::default();
        // 默认 avq=false, distance=l2 → 不再违反 avq_l2_conflict
        let result = engine.validate(&cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_avq_with_ip() {
        let engine = RuleEngine::default();
        let cfg = Config {
            avq: true,
            distance: "ip".to_string(),
            ..Default::default()
        };
        let result = engine.validate(&cfg);
        // avq + ip 不违反 avq_l2_conflict
        if let Err(e) = &result {
            assert!(!e.to_string().contains("avq_l2_conflict"));
        }
    }

    #[test]
    fn validate_no_avq_with_l2() {
        let engine = RuleEngine::default();
        let cfg = Config {
            avq: false,
            distance: "l2".to_string(),
            ..Default::default()
        };
        let result = engine.validate(&cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn gemm_threshold_violation() {
        let engine = RuleEngine::default();
        let cfg = Config {
            avq: false,
            distance: "l2".to_string(),
            gemm_path: true,
            candidate_count: 4,
            gemm_threshold: 8,
            ..Default::default()
        };
        let result = engine.validate(&cfg);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("gemm_threshold_check"));
    }

    #[test]
    fn gemm_threshold_ok() {
        let engine = RuleEngine::default();
        let cfg = Config {
            avq: false,
            distance: "l2".to_string(),
            gemm_path: true,
            candidate_count: 16,
            gemm_threshold: 8,
            ..Default::default()
        };
        let result = engine.validate(&cfg);
        assert!(result.is_ok());
    }

    // === Week 7-8 扩展规则测试 ===

    #[test]
    fn alpha_positive_violation() {
        let engine = RuleEngine::default();
        let cfg = Config {
            avq: false,
            distance: "l2".to_string(),
            alpha: 0.0,
            ..Default::default()
        };
        let result = engine.validate(&cfg);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("alpha_positive"));
    }

    #[test]
    fn alpha_negative_violation() {
        let engine = RuleEngine::default();
        let cfg = Config {
            avq: false,
            distance: "l2".to_string(),
            alpha: -1.0,
            ..Default::default()
        };
        let result = engine.validate(&cfg);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("alpha_positive"));
    }

    #[test]
    fn alpha_high_warning() {
        let engine = RuleEngine::default();
        let cfg = Config {
            avq: false,
            distance: "l2".to_string(),
            alpha: 1.2,
            ..Default::default()
        };
        // α=1.2 不违反 Error 规则，但触发 Warning
        assert!(engine.validate(&cfg).is_ok());
        let warnings = engine.check_warnings(&cfg);
        assert!(warnings.iter().any(|w| w.name == "alpha_high_warning"));
    }

    #[test]
    fn alpha_low_no_warning() {
        let engine = RuleEngine::default();
        let cfg = Config {
            avq: false,
            distance: "l2".to_string(),
            alpha: 0.5,
            ..Default::default()
        };
        assert!(engine.validate(&cfg).is_ok());
        let warnings = engine.check_warnings(&cfg);
        assert!(!warnings.iter().any(|w| w.name == "alpha_high_warning"));
    }

    #[test]
    fn codebook_k_not_power_of_two() {
        let engine = RuleEngine::default();
        let cfg = Config {
            avq: true,
            distance: "ip".to_string(),
            codebook_k: 255, // 不是 2 的幂次
            ..Default::default()
        };
        let result = engine.validate(&cfg);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("codebook_k_power_of_two"));
    }

    #[test]
    fn codebook_k_power_of_two_ok() {
        let engine = RuleEngine::default();
        let cfg = Config {
            avq: true,
            distance: "ip".to_string(),
            codebook_k: 256,
            ..Default::default()
        };
        let result = engine.validate(&cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn sub_dim_not_dividing_dim() {
        let engine = RuleEngine::default();
        let cfg = Config {
            avq: false,
            distance: "l2".to_string(),
            dim: 128,
            sub_dim: 7, // 128 % 7 != 0
            ..Default::default()
        };
        let result = engine.validate(&cfg);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("sub_dim_divides_dim"));
    }

    #[test]
    fn sub_dim_divides_dim_ok() {
        let engine = RuleEngine::default();
        let cfg = Config {
            avq: false,
            distance: "l2".to_string(),
            dim: 128,
            sub_dim: 8, // 128 % 8 == 0
            ..Default::default()
        };
        let result = engine.validate(&cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn top_n_less_than_k_violation() {
        let engine = RuleEngine::default();
        let cfg = Config {
            avq: false,
            distance: "l2".to_string(),
            k: 10,
            top_n: 5, // top_n < k
            ..Default::default()
        };
        let result = engine.validate(&cfg);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("top_n_ge_k"));
    }

    #[test]
    fn top_n_equal_k_ok() {
        let engine = RuleEngine::default();
        let cfg = Config {
            avq: false,
            distance: "l2".to_string(),
            k: 10,
            top_n: 10, // top_n == k
            ..Default::default()
        };
        let result = engine.validate(&cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn avq_alpha_out_of_range() {
        let engine = RuleEngine::default();
        let cfg = Config {
            avq: true,
            distance: "ip".to_string(),
            avq_alpha: 1.5, // > 1.0
            ..Default::default()
        };
        let result = engine.validate(&cfg);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("avq_alpha_range"));
    }

    #[test]
    fn avq_alpha_negative() {
        let engine = RuleEngine::default();
        let cfg = Config {
            avq: true,
            distance: "ip".to_string(),
            avq_alpha: -0.1,
            ..Default::default()
        };
        let result = engine.validate(&cfg);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("avq_alpha_range"));
    }

    #[test]
    fn avq_alpha_boundary_ok() {
        let engine = RuleEngine::default();
        // α=0.0 和 α=1.0 都是合法的
        for &alpha in &[0.0f32, 1.0] {
            let cfg = Config {
                avq: true,
                distance: "ip".to_string(),
                avq_alpha: alpha,
                ..Default::default()
            };
            let result = engine.validate(&cfg);
            assert!(result.is_ok(), "avq_alpha={} should be valid", alpha);
        }
    }

    #[test]
    fn avx512_dim_alignment_violation() {
        let engine = RuleEngine::default();
        let cfg = Config {
            avq: false,
            distance: "l2".to_string(),
            avx512: true,
            dim: 100, // 100 % 16 != 0
            ..Default::default()
        };
        let result = engine.validate(&cfg);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("avx512_dim_alignment"));
    }

    #[test]
    fn avx512_dim_alignment_ok() {
        let engine = RuleEngine::default();
        let cfg = Config {
            avq: false,
            distance: "l2".to_string(),
            avx512: true,
            dim: 128, // 128 % 16 == 0
            ..Default::default()
        };
        let result = engine.validate(&cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn is_power_of_two_check() {
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(is_power_of_two(256));
        assert!(!is_power_of_two(0));
        assert!(!is_power_of_two(3));
        assert!(!is_power_of_two(255));
    }

    // === DAG 冲突校验测试（设计文档：含 DAG 冲突校验）===

    #[test]
    fn dag_no_cycle_default_rules() {
        let engine = RuleEngine::default();
        assert!(engine.validate_dag().is_ok());
    }

    #[test]
    fn dag_cycle_detected() {
        let mut engine = RuleEngine::new_empty();
        engine.add_rule(
            Rule::new("rule_a", "A", |_| true, "ok")
                .with_depends_on(&["rule_b"])
        );
        engine.add_rule(
            Rule::new("rule_b", "B", |_| true, "ok")
                .with_depends_on(&["rule_a"])
        );
        assert!(engine.validate_dag().is_err());
    }

    #[test]
    fn dag_no_cycle_with_deps() {
        let mut engine = RuleEngine::new_empty();
        engine.add_rule(Rule::new("rule_a", "A", |_| true, "ok"));
        engine.add_rule(
            Rule::new("rule_b", "B", |_| true, "ok")
                .with_depends_on(&["rule_a"])
        );
        assert!(engine.validate_dag().is_ok());
    }
}
