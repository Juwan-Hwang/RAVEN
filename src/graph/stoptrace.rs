//! H40 学习式终止：逐跳特征/标签记录器（仅 diag_counters 诊断二进制启用，生产路径零开销）。
//! 每查询设置 GT（新 id 空间）后，sq4_expand_l0 每扩展一跳调用 `on_hop`，记录：
//! [qid, hop, gt_in_pool, gt_in_top20, gt_in_top40, features...]
use super::linear_pool::LinearPool;
use std::cell::RefCell;

pub const NF: usize = 16;
pub const NCOL: usize = 5 + NF;

pub struct Tracer {
    pub qid: u32,
    pub gt: Vec<u32>,
    pub rows: Vec<f32>,
    // per-query running state
    hop: u32,
    entry_d: f32,
    last_top10_sig: u64,
    stale10: u32,
    last_worst: f32,
    stale_w: u32,
    d10_hist: [f32; 8],
}

thread_local! {
    pub static TRACER: RefCell<Option<Tracer>> = const { RefCell::new(None) };
}

pub fn enable() { TRACER.with(|t| *t.borrow_mut() = Some(Tracer { qid: 0, gt: Vec::new(), rows: Vec::new(), hop: 0, entry_d: 0.0, last_top10_sig: 0, stale10: 0, last_worst: f32::MAX, stale_w: 0, d10_hist: [0.0; 8] })); }
pub fn take_rows() -> Vec<f32> { TRACER.with(|t| t.borrow_mut().as_mut().map(|x| std::mem::take(&mut x.rows)).unwrap_or_default()) }
pub fn begin_query(qid: u32, gt: &[u32]) {
    TRACER.with(|t| if let Some(x) = t.borrow_mut().as_mut() {
        x.qid = qid; x.gt.clear(); x.gt.extend_from_slice(gt); x.hop = 0; x.entry_d = 0.0;
        x.last_top10_sig = 0; x.stale10 = 0; x.last_worst = f32::MAX; x.stale_w = 0; x.d10_hist = [0.0; 8];
    });
}

#[inline(never)]
pub fn on_hop(pool: &LinearPool, inserted: u32, ins_top10: u32) {
    TRACER.with(|t| {
        let mut b = t.borrow_mut();
        let Some(x) = b.as_mut() else { return };
        let (data, cursor, ef) = pool.raw_view();
        let n = data.len();
        if n == 0 { return; }
        let id = |e: &(u32, f32)| e.0 & 0x7FFF_FFFF;
        let unc = |e: &(u32, f32)| (e.0 >> 31) == 0;
        if x.hop == 0 { x.entry_d = data[0].1.max(1e-12); }
        x.hop += 1;
        let k10 = 10.min(n);
        let d10 = data[k10 - 1].1.max(1e-12);
        let d1 = data[0].1;
        let worst = data[n - 1].1.max(1e-12);
        let mut sig: u64 = 0;
        for e in &data[..k10] { sig = sig.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(id(e) as u64 + 1); }
        if sig != x.last_top10_sig { x.stale10 = 0; x.last_top10_sig = sig; } else { x.stale10 += 1; }
        if worst != x.last_worst { x.stale_w = 0; x.last_worst = worst; } else { x.stale_w += 1; }
        let h = x.hop as usize;
        let d10_4 = x.d10_hist[(h + 4) % 8]; // value from hop h-4
        x.d10_hist[h % 8] = d10;
        let slope = if d10_4 > 0.0 { d10 / d10_4 } else { 1.0 };
        let lim = n.min(ef);
        let (mut unexp, mut unexp_lt10, mut unexp_lt12) = (0u32, 0u32, 0u32);
        let mut unexp_d: [f32; 3] = [f32::MAX; 3];
        let mut ui = 0usize;
        for e in &data[..lim] {
            if unc(e) {
                unexp += 1;
                if e.1 < d10 { unexp_lt10 += 1; }
                if e.1 < 1.2 * d10 { unexp_lt12 += 1; }
                if ui < 3 { unexp_d[ui] = e.1; ui += 1; }
            }
        }
        let next_r = if cursor < lim { data[cursor].1 / d10 } else { 9.0 };
        let r2 = if unexp_d[1] < f32::MAX { unexp_d[1] / d10 } else { 9.0 };
        let r3 = if unexp_d[2] < f32::MAX { unexp_d[2] / d10 } else { 9.0 };
        // GT counts
        let (mut g_all, mut g20, mut g40) = (0u32, 0u32, 0u32);
        for (r, e) in data.iter().enumerate() {
            if x.gt.contains(&id(e)) { g_all += 1; if r < 20 { g20 += 1; } if r < 40 { g40 += 1; } }
        }
        let f: [f32; NF] = [
            x.hop as f32, x.stale10 as f32, x.stale_w as f32, next_r, r2, r3,
            cursor as f32, unexp as f32, unexp_lt10 as f32, unexp_lt12 as f32,
            d1 / d10, d10 / worst, slope, x.entry_d / d10, inserted as f32, ins_top10 as f32,
        ];
        x.rows.extend_from_slice(&[x.qid as f32, x.hop as f32, g_all as f32, g20 as f32, g40 as f32]);
        x.rows.extend_from_slice(&f);
    });
}
