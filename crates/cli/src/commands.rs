use std::path::{Path, PathBuf};

/// Run the doctor command: check binary version, CUDA availability, and model files.
pub fn doctor(model_path: Option<&Path>) {
    println!("zipcode doctor\n");

    // Version check
    println!("Version: {}", env!("CARGO_PKG_VERSION"));
    println!();

    // CUDA check
    let cuda_ok = check_cuda();
    if cuda_ok {
        println!("  \u{2705} CUDA available");
    } else {
        println!("  \u{274c} CUDA not available (CPU inference only)");
    }

    // Model file check
    if let Some(path) = model_path {
        check_model_path(path);
    } else {
        // Check default model locations
        let default_dirs = default_model_dirs();
        let mut found_any = false;
        for dir in &default_dirs {
            if dir.exists() {
                let models = find_gguf_files(dir);
                if !models.is_empty() {
                    for model in &models {
                        println!("  \u{2705} Model found: {}", model.display());
                    }
                    found_any = true;
                }
            }
        }
        if !found_any {
            println!("  \u{274c} No model files found");
            println!(
                "    Searched: {}",
                default_dirs
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            println!("    Run: ./scripts/download_model.sh to download a model");
        }
    }

    println!();
    println!("Done.");
}

/// Check if CUDA is available by looking for libcuda.so or CUDA_PATH env var.
pub fn check_cuda() -> bool {
    // Check CUDA_PATH environment variable
    if std::env::var("CUDA_PATH").is_ok() {
        return true;
    }
    if std::env::var("CUDA_HOME").is_ok() {
        return true;
    }

    // Check for libcuda.so in common locations
    let cuda_libs = [
        "/usr/lib/x86_64-linux-gnu/libcuda.so",
        "/usr/lib/libcuda.so",
        "/usr/local/cuda/lib64/libcuda.so",
        "/usr/local/cuda/lib/libcuda.so",
    ];

    for lib in &cuda_libs {
        if Path::new(lib).exists() {
            return true;
        }
    }

    // Check via ldconfig
    if let Ok(output) = std::process::Command::new("ldconfig").arg("-p").output() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if stdout.contains("libcuda.so") {
            return true;
        }
    }

    false
}

fn check_model_path(path: &Path) {
    if path.exists() {
        let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let size_gb = size as f64 / (1024.0 * 1024.0 * 1024.0);
        println!(
            "  \u{2705} Model file: {} ({size_gb:.1} GB)",
            path.display()
        );
    } else {
        println!("  \u{274c} Model file not found: {}", path.display());
    }
}

fn default_model_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    // ~/.zipcode/models
    if let Some(home) = dirs::home_dir() {
        dirs.push(home.join(".zipcode/models"));
    }

    // ./models (relative to cwd)
    if let Ok(cwd) = std::env::current_dir() {
        dirs.push(cwd.join("models"));
    }

    dirs
}

fn find_gguf_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };

    entries
        .flatten()
        .filter_map(|e| {
            let path = e.path();
            if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
                Some(path)
            } else {
                None
            }
        })
        .collect()
}
