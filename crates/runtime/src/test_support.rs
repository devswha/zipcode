//! Shared test helpers for the runtime crate.
//!
//! Tests that mutate `ZIPCODE_SESSIONS_DIR` MUST hold [`SESSION_DIR_LOCK`]
//! while running, otherwise parallel tests race on the global env var and
//! `Session::path` resolution observes a foreign temp dir mid-test.

#![cfg(test)]

use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

use tempfile::TempDir;

/// Crate-wide mutex serialising every test that mutates
/// `ZIPCODE_SESSIONS_DIR`.  Tests in different modules MUST share this
/// single lock — separate per-module mutexes do not serialise the env var
/// because the env var is process-global.
pub(crate) static SESSION_DIR_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

/// Redirect session writes to a fresh temp dir, returning the [`TempDir`]
/// (which the caller must keep alive for the test's scope) and the held
/// [`SESSION_DIR_LOCK`] guard so the env var mutation cannot collide with
/// sibling tests.
pub(crate) fn with_test_session_dir() -> (TempDir, MutexGuard<'static, ()>) {
    let guard = SESSION_DIR_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let dir = TempDir::new().expect("tempdir");
    std::env::set_var("ZIPCODE_SESSIONS_DIR", dir.path());
    (dir, guard)
}
