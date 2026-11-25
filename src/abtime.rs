//! R66：进程内 A/B 计时统计（raven_eval `RAVEN_ABX`）。
//!
//! 同一进程、同一张图上建 A、B 两条搜索管线，按查询块（缺省 500 条）交替计时（块内先后按 rep/块/ef 轮换），
//! 每个 (rep, 块) 得到一对耗时 (tA, tB)。逐 ef 报告配对相对差 Δ = tB/tA − 1 的中位数，
//! 以及按 rep 整簇重抽样（cluster bootstrap，rep 内各块共享时间段，整簇抽更保守）得到的中位数 95% 置信区间。
//! 纯函数、确定性（splitmix64），可单元测试；不依赖搜索代码。

/// 一个 (rep, 块) 的配对耗时（秒或任意同单位）。
#[derive(Clone, Copy, Debug)]
pub struct Pair {
    pub rep: usize,
    pub ta: f64,
    pub tb: f64,
}

impl Pair {
    /// 相对差 tB/tA − 1（负 = B 更快）。
    pub fn delta(&self) -> f64 { self.tb / self.ta - 1.0 }
}

/// 逐 ef 汇总。
#[derive(Clone, Copy, Debug)]
pub struct AbStat {
    /// 配对相对差中位数
    pub median: f64,
    /// 95% CI（cluster bootstrap 百分位）
    pub lo: f64,
    pub hi: f64,
    /// 合并比 ΣtB/ΣtA − 1
    pub pooled: f64,
    pub n_pairs: usize,
    pub n_reps: usize,
    /// B 更快的配对占比
    pub frac_b_faster: f64,
}

impl AbStat {
    /// CI 是否包含 0
    pub fn ci_contains_zero(&self) -> bool { self.lo <= 0.0 && self.hi >= 0.0 }
    /// CI 半宽的最大值（|lo|、|hi| 较大者），用于"A/A 带比 ±x 窄"
    pub fn ci_abs_max(&self) -> f64 { self.lo.abs().max(self.hi.abs()) }
}

/// 中位数（偶数取中间两数均值）；空切片返回 NaN。会就地排序。
pub fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() { return f64::NAN; }
    v.sort_by(|a, b| a.total_cmp(b));
    let m = v.len() / 2;
    if v.len() % 2 == 1 { v[m] } else { 0.5 * (v[m - 1] + v[m]) }
}

/// 百分位（线性插值，p∈[0,1]）；v 须已排序。
pub fn percentile_sorted(v: &[f64], p: f64) -> f64 {
    if v.is_empty() { return f64::NAN; }
    let x = p.clamp(0.0, 1.0) * (v.len() - 1) as f64;
    let i = x.floor() as usize;
    let j = (i + 1).min(v.len() - 1);
    v[i] + (x - i as f64) * (v[j] - v[i])
}

/// 块内先跑哪一侧（0 = A 先，1 = B 先）：按 rep、块号、ef 序号轮换，避免固定先后带来的缓存/频率偏置。
pub fn first_side(rep: usize, block: usize, ef_idx: usize) -> usize { (rep + block + ef_idx) % 2 }

/// R75：B 侧管线怎么来（侧别偏差修正）。
/// - `Same`：只改运行期全局旋钮的 spec（`pfl:N`、`same`）——A、B 用**同一个** Prepared，只在每块前切旋钮；
///   两侧内存完全相同，物理放置偏差按构造为 0。
/// - `Sym { b_first }`：需要第二份 Prepared 的 spec（aa/pad/adc）——丢掉进程原管线，两侧都"重载同一图 + prepare"
///   同一代码路径各建一份；`b_first` 交换建造先后（RAVEN_ABX_ORDER=ba），两进程互换取平均可抵消先后放置效应。
/// - `Legacy`：R66–R74 旧行为（A = 进程原管线，B 后建），RAVEN_ABX_LEGACY=1，仅供复现偏差。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuildMode { Same, Sym { b_first: bool }, Legacy }

/// spec 是否只改运行期旋钮（不需要第二份 Prepared）。
pub fn runtime_only_spec(spec: &str) -> bool { spec == "same" || spec.starts_with("pfl:") }

/// 由 spec 与环境决定构造方式（纯函数，便于测试）。运行期旋钮 spec 永远 `Same`（即使 legacy=true）。
pub fn build_mode(spec: &str, legacy: bool, order_ba: bool) -> BuildMode {
    if runtime_only_spec(spec) { BuildMode::Same } else if legacy { BuildMode::Legacy } else { BuildMode::Sym { b_first: order_ba } }
}

/// 第 rep 轮里先建哪一侧的 searcher（0 = A）：逐 rep 交换，使 searcher 内部缓冲区（visited 等）的分配先后两侧各半。
pub fn searcher_first(rep: usize) -> usize { rep % 2 }

/// 解析 /proc/self/smaps 文本：与 [lo, hi) 相交的各 VMA 的 (Rss kB, AnonHugePages kB) 之和（整段 VMA 计入）。
pub fn smaps_huge(smaps: &str, lo: usize, hi: usize) -> (usize, usize) {
    let (mut rss, mut huge, mut hit) = (0usize, 0usize, false);
    for line in smaps.lines() {
        let mut it = line.split_whitespace();
        let first = match it.next() { Some(f) => f, None => continue };
        if let Some((a, b)) = first.split_once('-') {
            if let (Ok(a), Ok(b)) = (usize::from_str_radix(a, 16), usize::from_str_radix(b, 16)) {
                if !first.ends_with(':') { hit = a < hi && lo < b; continue; }
            }
        }
        if !hit { continue; }
        let val = || it.clone().next().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
        match first { "Rss:" => rss += val(), "AnonHugePages:" => huge += val(), _ => {} }
    }
    (rss, huge)
}

/// R75：先后顺序平衡的"对偶"配对。块交替计时里，先跑的一侧正好也是上一块最后跑的一侧（A B | B A | A B …），
/// 它连跑两块、缓存是热的；后跑的一侧紧跟另一条管线之后、工作集被挤出 → 逐块 Δ 呈 ±(先后效应) 双峰
/// （nytimes 两条管线 ±3…9%），双峰分布的中位数落在两峰之间的空隙里、对两峰形状的微小不对称极敏感 → 侧别偏差。
/// 把同一 rep 内相邻、先后相反的两块合成一个对偶（tA = tA₁+tA₂，tB = tB₁+tB₂），先后效应一阶抵消，分布单峰。
/// `firsts[i]` 是 `pairs[i]` 那块先跑的一侧（0 = A）；不成对的尾块丢弃。
pub fn couples(pairs: &[Pair], firsts: &[usize]) -> Vec<Pair> {
    assert_eq!(pairs.len(), firsts.len());
    let mut out = Vec::with_capacity(pairs.len() / 2);
    let mut i = 0;
    while i + 1 < pairs.len() {
        let (a, b) = (&pairs[i], &pairs[i + 1]);
        if a.rep == b.rep && firsts[i] != firsts[i + 1] {
            out.push(Pair { rep: a.rep, ta: a.ta + b.ta, tb: a.tb + b.tb });
            i += 2;
        } else { i += 1; }
    }
    out
}

fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// 汇总配对：中位数 + 按 rep 整簇 bootstrap 的 95% CI（n_boot 次，seed 固定 → 结果可复现）。
pub fn ab_stats(pairs: &[Pair], n_boot: usize, seed: u64) -> AbStat {
    let mut reps: Vec<usize> = pairs.iter().map(|p| p.rep).collect();
    reps.sort_unstable();
    reps.dedup();
    let clusters: Vec<Vec<f64>> = reps.iter().map(|&r| pairs.iter().filter(|p| p.rep == r).map(|p| p.delta()).collect()).collect();
    let mut all: Vec<f64> = pairs.iter().map(|p| p.delta()).collect();
    let n_pairs = all.len();
    let frac_b_faster = if n_pairs == 0 { f64::NAN } else { all.iter().filter(|&&d| d < 0.0).count() as f64 / n_pairs as f64 };
    let med = median(&mut all);
    let (sa, sb) = pairs.iter().fold((0.0, 0.0), |(a, b), p| (a + p.ta, b + p.tb));
    let pooled = if sa > 0.0 { sb / sa - 1.0 } else { f64::NAN };
    let nc = clusters.len();
    let (lo, hi) = if nc == 0 || n_boot == 0 { (f64::NAN, f64::NAN) } else {
        let mut s = seed ^ 0x5EED_AB66;
        let mut meds: Vec<f64> = Vec::with_capacity(n_boot);
        let mut buf: Vec<f64> = Vec::with_capacity(n_pairs);
        for _ in 0..n_boot {
            buf.clear();
            for _ in 0..nc { let c = (splitmix(&mut s) % nc as u64) as usize; buf.extend_from_slice(&clusters[c]); }
            meds.push(median(&mut buf));
        }
        meds.sort_by(|a, b| a.total_cmp(b));
        (percentile_sorted(&meds, 0.025), percentile_sorted(&meds, 0.975))
    };
    AbStat { median: med, lo, hi, pooled, n_pairs, n_reps: nc, frac_b_faster }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(s: &mut u64) -> f64 { (splitmix(s) >> 11) as f64 / (1u64 << 53) as f64 - 0.5 }

    #[test]
    fn median_and_percentile() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 2.0, 3.0]), 2.5);
        assert!(median(&mut []).is_nan());
        let v = [0.0, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(percentile_sorted(&v, 0.0), 0.0);
        assert_eq!(percentile_sorted(&v, 1.0), 4.0);
        assert!((percentile_sorted(&v, 0.5) - 2.0).abs() < 1e-12);
        assert!((percentile_sorted(&v, 0.125) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn first_side_alternates() {
        assert_eq!(first_side(0, 0, 0), 0);
        assert_eq!(first_side(0, 1, 0), 1);
        assert_eq!(first_side(1, 1, 0), 0);
        assert_eq!(first_side(0, 0, 1), 1);
        // 一个 rep 内（偶数块）A 先与 B 先各半
        let a_first = (0..20).filter(|&b| first_side(3, b, 2) == 0).count();
        assert_eq!(a_first, 10);
    }

    #[test]
    fn identical_arms_give_zero() {
        let pairs: Vec<Pair> = (0..10).flat_map(|r| (0..20).map(move |b| Pair { rep: r, ta: 1.0 + b as f64 * 0.01, tb: 1.0 + b as f64 * 0.01 })).collect();
        let s = ab_stats(&pairs, 500, 1);
        assert_eq!(s.median, 0.0);
        assert_eq!(s.lo, 0.0);
        assert_eq!(s.hi, 0.0);
        assert!(s.ci_contains_zero());
        assert_eq!(s.n_pairs, 200);
        assert_eq!(s.n_reps, 10);
        assert_eq!(s.pooled, 0.0);
    }

    #[test]
    fn aa_noise_ci_contains_zero_and_is_narrow() {
        // A/A：两侧同分布 ±5% 均匀噪声，外加每 rep 共同漂移（两侧同乘，配对后应抵消）
        let mut s = 7u64;
        let mut pairs = Vec::new();
        for r in 0..30 {
            let drift = 1.0 + 0.3 * noise(&mut s);
            for _ in 0..20 { pairs.push(Pair { rep: r, ta: drift * (1.0 + 0.1 * noise(&mut s)), tb: drift * (1.0 + 0.1 * noise(&mut s)) }); }
        }
        let st = ab_stats(&pairs, 2000, 42);
        assert!(st.ci_contains_zero(), "{st:?}");
        assert!(st.ci_abs_max() < 0.03, "{st:?}");
        assert!(st.median.abs() < 0.01, "{st:?}");
    }

    #[test]
    fn real_shift_detected() {
        // B 恒快 6%，±5% 噪声 + rep 漂移 → 中位约 −6%，CI 不含 0 且包含 −6%
        let mut s = 11u64;
        let mut pairs = Vec::new();
        for r in 0..20 {
            let drift = 1.0 + 0.3 * noise(&mut s);
            for _ in 0..20 { pairs.push(Pair { rep: r, ta: drift * (1.0 + 0.1 * noise(&mut s)), tb: 0.94 * drift * (1.0 + 0.1 * noise(&mut s)) }); }
        }
        let st = ab_stats(&pairs, 2000, 3);
        assert!(!st.ci_contains_zero(), "{st:?}");
        assert!(st.hi < 0.0 && st.lo < -0.06 && st.hi > -0.06, "{st:?}");
        assert!((st.median + 0.06).abs() < 0.015, "{st:?}");
        assert!(st.frac_b_faster > 0.85, "{st:?}");
    }

    #[test]
    fn bootstrap_is_deterministic() {
        let pairs: Vec<Pair> = (0..8).flat_map(|r| (0..5).map(move |b| Pair { rep: r, ta: 1.0, tb: 1.0 + ((r * 5 + b) % 7) as f64 * 0.01 - 0.03 })).collect();
        let a = ab_stats(&pairs, 300, 9);
        let b = ab_stats(&pairs, 300, 9);
        assert_eq!(a.lo, b.lo);
        assert_eq!(a.hi, b.hi);
    }
    #[test]
    fn r75_build_mode() {
        assert_eq!(build_mode("pfl:2", false, false), BuildMode::Same);
        assert_eq!(build_mode("pfl:1", true, true), BuildMode::Same);
        assert_eq!(build_mode("same", false, false), BuildMode::Same);
        assert_eq!(build_mode("aa", false, false), BuildMode::Sym { b_first: false });
        assert_eq!(build_mode("pad16", false, true), BuildMode::Sym { b_first: true });
        assert_eq!(build_mode("adc", true, false), BuildMode::Legacy);
        assert!(runtime_only_spec("pfl:2") && !runtime_only_spec("pad:16") && !runtime_only_spec("aa"));
        // searcher 先后逐 rep 交换，偶数 rep 数两侧各半
        assert_eq!((0..12).filter(|&r| searcher_first(r) == 0).count(), 6);
    }

    #[test]
    fn r75_smaps_parse() {
        let t = "7f0000000000-7f0000400000 rw-p 00000000 00:00 0 \nSize:               4096 kB\nRss:                4096 kB\nAnonHugePages:      2048 kB\nVmFlags: rd wr mr mw me ac hg\n7f0000400000-7f0000800000 rw-p 00000000 00:00 0 \nSize:               4096 kB\nRss:                1000 kB\nAnonHugePages:         0 kB\n55aa00000000-55aa00001000 r-xp 00000000 08:01 123 /bin/x\nRss:                   4 kB\nAnonHugePages:         0 kB\n";
        assert_eq!(smaps_huge(t, 0x7f0000000000, 0x7f0000001000), (4096, 2048));
        assert_eq!(smaps_huge(t, 0x7f00003ff000, 0x7f0000401000), (5096, 2048));
        assert_eq!(smaps_huge(t, 0x7f0000500000, 0x7f0000600000), (1000, 0));
        assert_eq!(smaps_huge(t, 0x1000, 0x2000), (0, 0));
    }

    #[test]
    fn r75_couples_cancel_order_effect() {
        // 真效应 0、先后效应：后跑的一侧慢 8%，外加逐块缓慢漂移；逐块中位数受双峰影响，对偶中位数 = 0
        let mut pairs = Vec::new(); let mut firsts = Vec::new();
        for rep in 0..12 { for blk in 0..20 { let f = first_side(rep, blk, 0); let base = 1.0 + 0.001 * (rep * 20 + blk) as f64;
            let (ta, tb) = if f == 0 { (base, base * 1.08) } else { (base * 1.08, base) };
            pairs.push(Pair { rep, ta, tb }); firsts.push(f); } }
        let cp = couples(&pairs, &firsts);
        assert_eq!(cp.len(), 12 * 10);
        let s = ab_stats(&cp, 500, 7);
        assert!(s.median.abs() < 1e-4 && s.lo.abs() < 1e-4 && s.hi.abs() < 1e-4, "{s:?}");
        // 对照：逐块统计是 ±8% 双峰（先跑 A 的块 Δ≈+8%，先跑 B 的块 Δ≈−7.4%）
        let n_far = pairs.iter().filter(|p| p.delta().abs() > 0.05).count();
        assert_eq!(n_far, pairs.len());
        // 真效应 +3%（B 慢）叠加同样的先后效应 → 对偶中位数 = +3%
        let p3: Vec<Pair> = pairs.iter().map(|p| Pair { rep: p.rep, ta: p.ta, tb: p.tb * 1.03 }).collect();
        let s3 = ab_stats(&couples(&p3, &firsts), 500, 7);
        assert!((s3.median - 0.03).abs() < 1e-4, "{s3:?}");
        // 跨 rep 或同先后的相邻块不配；奇数尾块丢弃
        let q = vec![Pair { rep: 0, ta: 1.0, tb: 1.0 }, Pair { rep: 1, ta: 1.0, tb: 1.0 }, Pair { rep: 1, ta: 1.0, tb: 2.0 }];
        assert_eq!(couples(&q, &[0, 1, 0]).len(), 1);
        assert_eq!(couples(&q, &[0, 0, 0]).len(), 0);
    }

}
