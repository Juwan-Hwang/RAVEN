//! 最小验证：prune_with_jumps 是否真的产出跳跃边？
//!
//! 用法：cargo run --release --bin jump_verify

use raven::graph::robust_prune::{prune_dispatch, PruneStrategy};

fn main() {
    let dim = 2;
    let vectors: Vec<f32> = vec![
        0.0, 0.0,   // node 0 (query)
        1.0, 0.0,   // node 1: 近, 0°
        0.9, 0.1,   // node 2: 近, ~6°
        0.8, 0.2,   // node 3: 近, ~14°
        2.0, 2.0,   // node 4: 中, 45°
        1.5, 1.5,   // node 5: 中, 45° (被4遮挡)
        5.0, -1.0,  // node 6: 远, ~-11°
        5.0, 1.0,   // node 7: 远, ~11°
        4.0, 3.0,   // node 8: 远, ~37°
        3.0, 4.0,   // node 9: 远, ~53°
        -3.0, 4.0,  // node 10: 远, ~127°
        -4.0, 3.0,  // node 11: 远, ~143°
        -5.0, 0.0,  // node 12: 远, 180°
        0.0, 5.0,   // node 13: 远, 90°
        0.0, -5.0,  // node 14: 远, -90°
    ];

    let candidates: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
    let query_node = 0u32;
    let r_max = 10usize;

    println!("=== 验证 prune_dispatch 是否读取环境变量 ===\n");

    // 不设置环境变量
    std::env::remove_var("RAVEN_N_JUMPS");
    let result_no_env = prune_dispatch(
        PruneStrategy::DirectionalPrune,
        &candidates, query_node, &vectors, dim,
        1.2, r_max, false,
    );
    println!("无环境变量: {} 邻居 → {:?}", result_no_env.len(), result_no_env);

    // n_jumps=2
    std::env::set_var("RAVEN_N_JUMPS", "2");
    let result_j2 = prune_dispatch(
        PruneStrategy::DirectionalPrune,
        &candidates, query_node, &vectors, dim,
        1.2, r_max, false,
    );
    println!("RAVEN_N_JUMPS=2: {} 邻居 → {:?}", result_j2.len(), result_j2);

    // n_jumps=4
    std::env::set_var("RAVEN_N_JUMPS", "4");
    let result_j4 = prune_dispatch(
        PruneStrategy::DirectionalPrune,
        &candidates, query_node, &vectors, dim,
        1.2, r_max, false,
    );
    println!("RAVEN_N_JUMPS=4: {} 邻居 → {:?}", result_j4.len(), result_j4);

    // n_jumps=8
    std::env::set_var("RAVEN_N_JUMPS", "8");
    let result_j8 = prune_dispatch(
        PruneStrategy::DirectionalPrune,
        &candidates, query_node, &vectors, dim,
        1.2, r_max, false,
    );
    println!("RAVEN_N_JUMPS=8: {} 邻居 → {:?}", result_j8.len(), result_j8);

    println!("\n=== 跳跃边增量 ===");
    println!("n_jumps=0 → 2: +{} 邻居", result_j2.len() as i32 - result_no_env.len() as i32);
    println!("n_jumps=0 → 4: +{} 邻居", result_j4.len() as i32 - result_no_env.len() as i32);
    println!("n_jumps=0 → 8: +{} 邻居", result_j8.len() as i32 - result_no_env.len() as i32);

    // 找出跳跃边
    let base_set: std::collections::HashSet<u32> = result_no_env.iter().copied().collect();
    println!("\nn_jumps=8 新增的跳跃边:");
    for &id in &result_j8 {
        if !base_set.contains(&id) {
            let v = &vectors[id as usize * dim..(id as usize + 1) * dim];
            let dist = (v[0].powi(2) + v[1].powi(2)).sqrt();
            let angle = v[1].atan2(v[0]).to_degrees();
            println!("  node {}: 距离={:.2}, 角度={:.1}°", id, dist, angle);
        }
    }

    println!("\n=== 结论 ===");
    if result_j4.len() > result_no_env.len() {
        println!("✅ 环境变量生效：跳跃边正在被添加");
    } else {
        println!("❌ 环境变量未生效：n_jumps=4 和 baseline 邻居数相同 ({})", result_no_env.len());
        println!("   → 跳跃边从未被添加，P12 实验结论无效！");
    }
}
