pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub const LONG_VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (",
    env!("ZIPCODE_BUILD_GIT"),
    env!("ZIPCODE_BUILD_DIRTY"),
    ", build ",
    env!("ZIPCODE_BUILD_EPOCH"),
    ")"
);

#[must_use]
pub const fn build_label() -> &'static str {
    LONG_VERSION
}

#[must_use]
pub const fn git_label() -> &'static str {
    concat!(env!("ZIPCODE_BUILD_GIT"), env!("ZIPCODE_BUILD_DIRTY"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_version_contains_package_version_and_build() {
        assert!(LONG_VERSION.contains(VERSION));
        assert!(LONG_VERSION.contains("build "));
    }

    #[test]
    fn git_label_contains_hash_or_unknown() {
        assert!(!git_label().is_empty());
    }
}
