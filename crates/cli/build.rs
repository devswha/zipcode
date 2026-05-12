use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=../../.git/HEAD");

    let git_hash =
        git_output(&["rev-parse", "--short=12", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    let dirty = tracked_worktree_is_dirty();
    let build_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs());

    println!("cargo:rustc-env=ZIPCODE_BUILD_GIT={git_hash}");
    println!(
        "cargo:rustc-env=ZIPCODE_BUILD_DIRTY={}",
        if dirty { "+dirty" } else { "" }
    );
    println!("cargo:rustc-env=ZIPCODE_BUILD_EPOCH={build_epoch}");
}

fn git_output(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn git_status_success(args: &[&str]) -> bool {
    Command::new("git")
        .args(args)
        .status()
        .is_ok_and(|status| status.success())
}

fn tracked_worktree_is_dirty() -> bool {
    !git_status_success(&["diff", "--quiet", "--ignore-submodules", "--"])
        || !git_status_success(&["diff", "--cached", "--quiet", "--ignore-submodules", "--"])
}
