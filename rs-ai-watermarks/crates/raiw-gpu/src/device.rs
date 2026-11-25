//! Compute device selection mirroring the Python `device` parameter.
//!
//! `auto` takes the first GPU adapter that survives a probe computation and falls
//! back to the pure-Rust CPU backend; `cpu`, `cuda` and `mps` are strict requests.

use std::fmt;
use std::panic::{self, AssertUnwindSafe};
use std::str::FromStr;

use burn::tensor::{Device, Tensor};

use crate::error::{GpuError, Result};

/// The device a caller asked for, as spelled by the Python `device` option.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceRequest {
    #[default]
    Auto,
    Cpu,
    /// A discrete compute GPU: native CUDA/ROCm when compiled in, else a wgpu discrete adapter.
    Cuda,
    /// Apple GPU through Metal.
    Mps,
}

impl DeviceRequest {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Mps => "mps",
        }
    }
}

impl FromStr for DeviceRequest {
    type Err = GpuError;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" | "" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "cuda" | "gpu" | "vulkan" | "rocm" => Ok(Self::Cuda),
            "mps" | "metal" => Ok(Self::Mps),
            _ => Err(GpuError::UnknownDevice(value.to_owned())),
        }
    }
}

impl fmt::Display for DeviceRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Which burn runtime executes the kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Pure-Rust CPU backend (`burn-flex`).
    Cpu,
    /// wgpu: Vulkan, Metal or DX12 depending on the platform.
    Wgpu,
    Cuda,
    Rocm,
}

impl Backend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Wgpu => "wgpu",
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
        }
    }
}

/// A probed, ready-to-use burn device plus the adapter it runs on.
#[derive(Debug, Clone)]
pub struct ComputeDevice {
    inner: Device,
    backend: Backend,
    adapter: String,
}

impl ComputeDevice {
    /// The always-available CPU fallback.
    pub fn cpu() -> Self {
        Self {
            inner: Device::flex(),
            backend: Backend::Cpu,
            adapter: "cpu (burn-flex)".to_owned(),
        }
    }

    /// Resolve a request to a working device; `auto` never fails.
    pub fn select(request: DeviceRequest) -> Result<Self> {
        let selected = match request {
            DeviceRequest::Cpu => return Ok(Self::cpu()),
            DeviceRequest::Auto => {
                return Ok(candidates(GpuClass::Any)
                    .into_iter()
                    .find_map(probe)
                    .unwrap_or_else(Self::cpu));
            }
            DeviceRequest::Cuda => candidates(GpuClass::Discrete).into_iter().find_map(probe),
            DeviceRequest::Mps => candidates(GpuClass::Apple).into_iter().find_map(probe),
        };
        selected.ok_or_else(|| GpuError::DeviceUnavailable {
            requested: request.to_string(),
            reason: "no compiled-in backend produced a working adapter".to_owned(),
        })
    }

    /// Parse and resolve a Python-style device string.
    pub fn from_name(name: &str) -> Result<Self> {
        Self::select(name.parse()?)
    }

    pub fn burn(&self) -> &Device {
        &self.inner
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Human-readable adapter name, e.g. `AMD Radeon RX 5500 XT (RADV NAVI14)`.
    pub fn adapter(&self) -> &str {
        &self.adapter
    }

    pub fn is_gpu(&self) -> bool {
        self.backend != Backend::Cpu
    }

    /// Block until queued kernels finish (for timing).
    pub fn sync(&self) -> Result<()> {
        self.inner
            .sync()
            .map_err(|err| GpuError::Readback(err.to_string()))
    }
}

impl fmt::Display for ComputeDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.adapter, self.backend.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuClass {
    Any,
    Discrete,
    Apple,
}

struct Candidate {
    device: Device,
    backend: Backend,
}

#[allow(clippy::vec_init_then_push, unused_mut)]
fn candidates(class: GpuClass) -> Vec<Candidate> {
    let mut out = Vec::new();
    if class != GpuClass::Apple {
        #[cfg(feature = "cuda")]
        out.push(Candidate {
            device: Device::cuda(0),
            backend: Backend::Cuda,
        });
        #[cfg(feature = "rocm")]
        out.push(Candidate {
            device: Device::rocm(0),
            backend: Backend::Rocm,
        });
    }
    #[cfg(feature = "wgpu")]
    {
        use burn::tensor::DeviceKind;
        let kinds: &[DeviceKind] = match class {
            GpuClass::Any => &[
                DeviceKind::DefaultDevice,
                DeviceKind::DiscreteGpu(0),
                DeviceKind::IntegratedGpu(0),
            ],
            GpuClass::Discrete => &[DeviceKind::DiscreteGpu(0)],
            GpuClass::Apple if cfg!(target_os = "macos") => {
                &[DeviceKind::DefaultDevice, DeviceKind::IntegratedGpu(0)]
            }
            GpuClass::Apple => &[],
        };
        out.extend(kinds.iter().map(|kind| Candidate {
            device: wgpu_device(kind.clone()),
            backend: Backend::Wgpu,
        }));
    }
    out
}

#[cfg(all(feature = "wgpu", feature = "vulkan"))]
fn wgpu_device(kind: burn::tensor::DeviceKind) -> Device {
    Device::vulkan(kind)
}

#[cfg(all(feature = "wgpu", not(feature = "vulkan")))]
fn wgpu_device(kind: burn::tensor::DeviceKind) -> Device {
    Device::wgpu(kind)
}

/// Run a tiny kernel on the candidate; adapter creation failures panic inside burn.
fn probe(candidate: Candidate) -> Option<ComputeDevice> {
    let Candidate { device, backend } = candidate;
    let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
        let values = Tensor::<1>::from_floats([1.0_f32, 2.0, 3.0], &device)
            .mul_scalar(2.0)
            .try_into_data()
            .ok()?;
        let values = values.try_into_vec_as::<f32>().ok()?;
        (values == [2.0, 4.0, 6.0]).then(|| adapter_name(&device))
    }));
    match outcome {
        Ok(Some(Some(adapter))) => {
            tracing::info!(adapter = %adapter, backend = backend.as_str(), "selected burn device");
            Some(ComputeDevice {
                inner: device,
                backend,
                adapter,
            })
        }
        Ok(Some(None)) => {
            tracing::debug!(backend = backend.as_str(), "skipping software adapter");
            None
        }
        Ok(None) | Err(_) => {
            tracing::debug!(backend = backend.as_str(), "device probe failed");
            None
        }
    }
}

/// The adapter name, or `None` for software rasterizers that are no real GPU.
#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
fn adapter_name(device: &Device) -> Option<String> {
    match device.identity() {
        Some(identity) if identity.physical.is_some() => Some(identity.name),
        Some(_) => None,
        None => Some(format!("{device:?}")),
    }
}

#[cfg(not(any(feature = "wgpu", feature = "cuda", feature = "rocm")))]
fn adapter_name(device: &Device) -> Option<String> {
    Some(format!("{device:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_python_device_names() {
        assert_eq!("auto".parse::<DeviceRequest>().unwrap(), DeviceRequest::Auto);
        assert_eq!("CPU".parse::<DeviceRequest>().unwrap(), DeviceRequest::Cpu);
        assert_eq!("cuda".parse::<DeviceRequest>().unwrap(), DeviceRequest::Cuda);
        assert_eq!("mps".parse::<DeviceRequest>().unwrap(), DeviceRequest::Mps);
        assert!("tpu".parse::<DeviceRequest>().is_err());
    }

    #[test]
    fn cpu_is_always_available() {
        let device = ComputeDevice::select(DeviceRequest::Cpu).unwrap();
        assert!(!device.is_gpu());
        let doubled = Tensor::<1>::from_floats([1.5_f32], device.burn()).mul_scalar(2.0);
        assert_eq!(doubled.into_data().try_into_vec_as::<f32>().unwrap(), vec![3.0]);
    }

    #[test]
    fn auto_never_fails() {
        let device = ComputeDevice::select(DeviceRequest::Auto).unwrap();
        assert!(!device.adapter().is_empty());
    }
}
