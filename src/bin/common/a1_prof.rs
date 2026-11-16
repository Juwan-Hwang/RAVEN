//! A1（第 12 轮）：精简路径（sq4_expand）暖缓存分段剖析——轨迹回放法。
//! diag 构建用 RAVEN_A1_DUMP=<file> 记录轨迹；release 构建用 RAVEN_A1_REPLAY=<file> 回放计时。
//! 回放保持轨迹逐位相同（弹出序列、邻居集合、距离均与记录一致，全开模式会校验），
//! 通过开关逐段去掉 visited 过滤 / SQ4 距离核 / 池插入弹出 / 预取，得到各段边际成本（暖缓存：每查询先回放一遍不计时）。
use raven::graph::linear_pool::LinearPool;
use raven::graph::vamana::{A1_BEGIN, A1_END, A1_HOP};
use raven::memory::VisitedTracker;
use std::time::Instant;

pub struct Hop { pub node: u32, pub ids: Vec<u32>, pub ds: Vec<f32> }
pub struct QTrace { pub seeds: Vec<(u32, f32)>, pub vis: Vec<u32>, pub hops: Vec<Hop> }

pub fn parse(t: &[u32]) -> Vec<QTrace> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < t.len() {
        assert_eq!(t[i], A1_BEGIN); i += 1;
        let n = t[i] as usize; i += 1;
        let seeds = (0..n).map(|k| (t[i + 2 * k] & 0x7FFF_FFFF, f32::from_bits(t[i + 2 * k + 1]))).collect(); i += 2 * n;
        i += 1; // cursor
        let nv = t[i] as usize; i += 1;
        let vis = t[i..i + nv].to_vec(); i += nv;
        let mut hops = Vec::new();
        while t[i] != A1_END {
            assert_eq!(t[i], A1_HOP); let node = t[i + 1]; i += 2;
            let (mut ids, mut ds) = (Vec::new(), Vec::new());
            while t[i] != A1_HOP && t[i] != A1_END { ids.push(t[i]); ds.push(f32::from_bits(t[i + 1])); i += 2; }
            hops.push(Hop { node, ids, ds });
        }
        i += 1;
        out.push(QTrace { seeds, vis, hops });
    }
    out
}

pub const V: u32 = 1; // visited 过滤（读邻接 + 位图）
pub const D: u32 = 2; // SQ4 距离核
pub const I: u32 = 4; // 池插入 + 弹出
pub const P: u32 = 8; // 预取（邻接 / visited / 码）
pub const T: u32 = 16; // 不算距离时仍读码首字节（保留访存，冷回放用）
pub const O: u32 = 32; // H134 oracle：查询开始时把全部将被扩展节点的邻接行预取到 L2（完美跳预测上界）
pub const C: u32 = 64; // H134 oracle：查询开始时把全部将被取的码行预取到 L2（完美轨迹预测上界）

#[inline(never)]
pub fn replay(q: &QTrace, qc: &[u8], p: &raven::pipeline::Prepared, ef: usize, vis: &mut VisitedTracker, pool: &mut LinearPool, f: u32, check: bool, buf: &mut [u32; 128]) -> f32 {
    let sq4 = &p.sq4; let st = p.graph.storage(); let sd = p.sd; let po = p.po;
    let mut sink = 0f32;
    vis.reset();
    for &v in &q.vis { vis.visit(v); }
    pool.reset_for(ef);
    for &(v, d) in &q.seeds { pool.insert(v, d); }
    let eff = |len: usize| if sd > 0 { sd.min(len) } else { len };
    if f & O != 0 {
        let lines = (eff(st.r_max()) * 4).div_ceil(64);
        for h in &q.hops { let ptr = st.main_block().as_ptr().wrapping_add(h.node as usize * st.r_max()) as *const i8; unsafe { for l in 0..lines { raven::arch::prefetch_l2(ptr.add(l * 64)); } } }
    }
    if f & C != 0 { for h in &q.hops { for &v in &h.ids { unsafe { raven::arch::prefetch_l2(sq4.code(v as usize).as_ptr() as *const i8); } } } }
    for (hi, h) in q.hops.iter().enumerate() {
        let node = if f & I != 0 { let n = pool.pop().unwrap().0; if check { assert_eq!(n, h.node, "pop mismatch hop {hi}"); } n } else { h.node };
        if f & P != 0 {
            let nx = if f & I != 0 { pool.peek_unchecked().map(|x| x.0) } else { q.hops.get(hi + 1).map(|x| x.node) };
            if let Some(nn) = nx {
                let ptr = st.main_block().as_ptr().wrapping_add(nn as usize * st.r_max()) as *const i8;
                let lines = (eff(st.r_max()) * 4).div_ceil(64);
                unsafe { for l in 0..lines { raven::arch::prefetch_l1(ptr.add(l * 64)); } }
            }
        }
        let (ids, n): (&[u32], usize) = if f & V != 0 {
            let nb = st.neighbors(node); let el = eff(nb.len());
            if f & P != 0 { for &v in nb.iter().take(po.min(el)) { unsafe { vis.prefetch(v); } } }
            let mut es = 0usize;
            for (i, &v) in nb.iter().take(el).enumerate() {
                if f & P != 0 && i + po < el { unsafe { vis.prefetch(nb[i + po]); } }
                let is_new = unsafe { vis.visit_unchecked_branchless(v) };
                unsafe { *buf.get_unchecked_mut(es) = v; }
                es += is_new as usize;
            }
            if check { assert_eq!(&buf[..es], &h.ids[..], "visited mismatch hop {hi}"); }
            (&buf[..], es)
        } else { (&h.ids[..], h.ids.len()) };
        if f & P != 0 { for &v in ids.iter().take(po.min(n)) { unsafe { raven::arch::prefetch_l1(sq4.code(v as usize).as_ptr() as *const i8); } } }
        for i in 0..n {
            if f & P != 0 && i + po < n { unsafe { raven::arch::prefetch_l1(sq4.code(ids[i + po] as usize).as_ptr() as *const i8); } }
            let v = ids[i];
            let d = if f & D != 0 { let d = unsafe { sq4.distance_raw_unchecked(qc, v as usize) }; if check { assert_eq!(d, h.ds[i]); } d } else { h.ds[i] };
            if f & D == 0 && f & T != 0 { sink += unsafe { *sq4.code_unchecked(v as usize).get_unchecked(0) } as f32; }
            if f & I != 0 { pool.insert(v, d); } else { sink += d; }
        }
    }
    sink
}

pub fn encode_query(p: &raven::pipeline::Prepared, q: &[f32]) -> Vec<u8> {
    let mut qc = vec![0u8; p.sq4.code_bytes];
    if p.perm.is_empty() { p.sq4.params.encode_into(q, &mut qc); } else {
        let b: Vec<f32> = p.perm.iter().map(|&d| q[d]).collect(); p.sq4.params.encode_into(&b, &mut qc);
    }
    qc
}

/// 距离核拆分微基准（暖：对每个查询的被取码集合循环计时）
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub unsafe fn kern_variant(qa: &[u8], qb: &[u8], mode: u32) -> u32 {
    use std::arch::aarch64::*;
    let n = qa.len(); let (pa, pb) = (qa.as_ptr(), qb.as_ptr());
    let mask = vdupq_n_u8(0x0F);
    let (mut a0, mut a1) = (vdupq_n_u32(0), vdupq_n_u32(0));
    for c in 0..n / 16 {
        let a = vld1q_u8(pa.add(c * 16)); let b = vld1q_u8(pb.add(c * 16));
        match mode {
            0 => { // 完整：解码（and/shr）+ abd + udot
                let dl = vabdq_u8(vandq_u8(a, mask), vandq_u8(b, mask)); let dh = vabdq_u8(vshrq_n_u8::<4>(a), vshrq_n_u8::<4>(b));
                a0 = vdotq_u32(a0, dl, dl); a1 = vdotq_u32(a1, dh, dh);
            }
            1 => { // 无解码：字节 abd + udot（成本参照）
                let d = vabdq_u8(a, b); a0 = vdotq_u32(a0, d, d);
            }
            _ => { // 只读：load + 累加
                a0 = vdotq_u32(a0, a, b);
            }
        }
    }
    vaddvq_u32(vaddq_u32(a0, a1))
}

pub fn run(p: &raven::pipeline::Prepared, path: &str, queries: &[f32], dim: usize, ef: usize, reps: usize) {
    let raw = std::fs::read(path).expect("trace file");
    let t: Vec<u32> = raw.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let qs = parse(&t);
    let nq = qs.len();
    let qcs: Vec<Vec<u8>> = (0..nq).map(|i| encode_query(p, &queries[i * dim..(i + 1) * dim])).collect();
    let n = p.graph.len();
    let mut vis = VisitedTracker::new(n, ef);
    let mut pool = LinearPool::new(ef);
    let mut buf = [0u32; 128];
    let hops: usize = qs.iter().map(|q| q.hops.len()).sum();
    let codes: usize = qs.iter().map(|q| q.hops.iter().map(|h| h.ids.len()).sum::<usize>()).sum();
    eprintln!("A1 trace: {nq} queries, hops/q {:.2}, codes/q {:.1}, seeds/q {:.1}, nav-visited/q {:.1}", hops as f64 / nq as f64, codes as f64 / nq as f64,
        qs.iter().map(|q| q.seeds.len()).sum::<usize>() as f64 / nq as f64, qs.iter().map(|q| q.vis.len()).sum::<usize>() as f64 / nq as f64);
    // 校验：全开回放与记录逐位一致
    for (i, q) in qs.iter().enumerate() { replay(q, &qcs[i], p, ef, &mut vis, &mut pool, V | D | I | P, true, &mut buf); }
    eprintln!("A1 check: full replay matches trace (pops, visited lists, distances)");
    // 真实搜索暖计时（同一批查询，每查询先跑一遍）
    let mut sx = p.searcher(ef);
    let cold = std::env::var("RAVEN_A1_COLD").map(|v| v == "1").unwrap_or(false);
    let modes: Vec<(&str, u32)> = if cold { vec![("full", V | D | I | P), ("-I", V | D | P), ("-D+T", V | I | P | T), ("-V", D | I | P), ("-V-I", D | P), ("P+T", P | T), ("V+P+T", V | P | T), ("T", T), ("full-P", V | D | I), ("full+O", V | D | I | P | O), ("full+C", V | D | I | P | C), ("full+O+C", V | D | I | P | O | C), ("T+O+C", T | O | C), ("V+P+T+O+C", V | P | T | O | C), ("-D-I+O+C", V | P | T | O | C)] } else { vec![("full", V | D | I | P), ("-P", V | D | I), ("-V", D | I | P), ("-D", V | I | P), ("-I", V | D | P), ("only-V", V), ("only-D", D), ("only-I", I), ("only-P", P), ("none", 0), ("noP:V+D", V | D), ("noP:D+I", D | I)] };
    let mut res: Vec<Vec<f64>> = vec![Vec::new(); modes.len() + 1];
    let mut sink = 0f32;
    for _ in 0..reps {
        // 真实 search_sq4（暖）
        let mut tt = 0f64;
        for i in 0..nq { let q = &queries[i * dim..(i + 1) * dim]; if !cold { sink += sx.search_sq4(q, 10).len() as f32; } let t0 = Instant::now(); sink += sx.search_sq4(q, 10).len() as f32; tt += t0.elapsed().as_secs_f64(); }
        res[0].push(tt / nq as f64 * 1e9);
        for (mi, &(_, f)) in modes.iter().enumerate() {
            let mut tt = 0f64;
            for (i, q) in qs.iter().enumerate() {
                if !cold { sink += replay(q, &qcs[i], p, ef, &mut vis, &mut pool, V | D | I | P, false, &mut buf); } // 暖：用全开回放把工作集拉进 cache
                let t0 = Instant::now();
                sink += replay(q, &qcs[i], p, ef, &mut vis, &mut pool, f, false, &mut buf);
                tt += t0.elapsed().as_secs_f64();
            }
            res[mi + 1].push(tt / nq as f64 * 1e9);
        }
    }
    let med = |v: &Vec<f64>| { let mut x = v.clone(); x.sort_by(|a, b| a.total_cmp(b)); x[x.len() / 2] };
    eprintln!("A1 cold={cold} ns/query (median of {reps}): real search_sq4 = {:.0}", med(&res[0]));
    let full = med(&res[1]);
    for (mi, &(name, _)) in modes.iter().enumerate() {
        let m = med(&res[mi + 1]);
        eprintln!("A1 replay {name:>8}: {m:8.0} ns/q  ({:+.0} vs full; per hop {:.1} ns; per code {:.2} ns)", m - full, m / hops as f64 * nq as f64, m / codes as f64 * nq as f64);
    }
    // 距离核拆分（暖：每查询的码集合连续算 4 遍，计后 3 遍）
    #[cfg(target_arch = "aarch64")]
    {
        let mut out = [0f64; 4];
        let mut acc = 0u32;
        for (mi, mode) in [0u32, 1, 2, 9].iter().enumerate() {
            let mut tt = 0f64; let mut cnt = 0usize;
            for (i, q) in qs.iter().enumerate() {
                let qc = &qcs[i];
                for rep in 0..4 {
                    let t0 = Instant::now();
                    for h in &q.hops { for &v in &h.ids {
                        let c = unsafe { p.sq4.code_unchecked(v as usize) };
                        acc = acc.wrapping_add(if *mode == 9 { unsafe { raven::quant::sq4::l2_sq4_raw_neon(qc, c) as u32 } } else { unsafe { kern_variant(qc, c, *mode) } });
                        if rep > 0 { cnt += 1; }
                    } }
                    if rep > 0 { tt += t0.elapsed().as_secs_f64(); }
                }
            }
            out[mi] = tt / cnt as f64 * 1e9;
        }
        eprintln!("A1 kernel ns/code (warm, throughput, not inlined): full={:.2} no-decode={:.2} load+dot={:.2} | production l2_sq4_raw_neon={:.2} (acc {acc})", out[0], out[1], out[2], out[3]);
    }
    eprintln!("A1 sink {sink}");
}
