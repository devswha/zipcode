use candle_core::Device;
use tracing::info;

/// Pick the best available compute device (CUDA GPU 0 if available, otherwise CPU).
pub fn select_device() -> Device {
    match Device::new_cuda(0) {
        Ok(device) => {
            info!("Using CUDA device 0");
            device
        }
        Err(e) => {
            info!("CUDA not available ({e}), falling back to CPU");
            Device::Cpu
        }
    }
}

/// Return a human-readable label for the device (`"CUDA GPU"` or `"CPU"`).
#[must_use]
pub fn device_info(device: &Device) -> String {
    match device {
        Device::Cpu => "CPU".to_string(),
        Device::Cuda(_) => "CUDA GPU".to_string(),
        Device::Metal(_) => "Unknown device".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_device_returns_valid() {
        let device = select_device();
        let info = device_info(&device);
        assert!(!info.is_empty());
    }
}
