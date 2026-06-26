//! build.rs — 编译时注入版本元信息
//!
//! 捕获 git commit hash + 构建时间戳，写入环境变量，
//! 供所有二进制通过 env!() / option_env!() 读取。
//! 用户可在终端输出中一眼分辨版本，避免误跑旧代码。

use std::process::Command;

fn main() {
    // 重新构建时机：git HEAD 变化或 build.rs 变化
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=.git/HEAD");

    // ── git commit short hash ──
    let git_hash = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    // ── git dirty 标记（有未提交改动则加 +） ──
    let git_dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);
    let git_hash_full = if git_dirty {
        format!("{}+dirty", git_hash)
    } else {
        git_hash
    };

    // ── 构建时间戳 ──
    let build_ts = Command::new("git")
        .args(["log", "-1", "--format=%cd", "--date=format:%Y-%m-%d_%H:%M:%S"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=RAVEN_GIT_HASH={}", git_hash_full);
    println!("cargo:rustc-env=RAVEN_BUILD_TS={}", build_ts);
}
