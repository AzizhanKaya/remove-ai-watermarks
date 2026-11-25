//! Host buffer <-> device tensor transfers.

use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::error::{GpuError, Result};

/// Validate a nonempty, finite mono buffer.
pub fn check_mono(samples: &[f32], what: &str) -> Result<()> {
    if samples.is_empty() {
        return Err(GpuError::Input(format!("{what} samples must be nonempty")));
    }
    if !samples.iter().all(|x| x.is_finite()) {
        return Err(GpuError::Input(format!("{what} samples must be finite")));
    }
    Ok(())
}

/// Upload mono samples as a `[1, 1, samples]` batch.
pub fn mono_tensor(samples: &[f32], device: &Device) -> Tensor<3> {
    Tensor::from_data(TensorData::new(samples.to_vec(), [1, 1, samples.len()]), device)
}

/// Read a float tensor back as a flat row-major buffer.
pub fn read_f32<const D: usize>(tensor: Tensor<D>) -> Result<Vec<f32>> {
    let data = tensor
        .try_into_data()
        .map_err(|err| GpuError::Readback(err.to_string()))?;
    data.try_into_vec_as::<f32>()
        .map_err(|err| GpuError::Readback(format!("{err:?}")))
}

/// Read a float tensor back and reject NaN/inf.
pub fn read_finite<const D: usize>(tensor: Tensor<D>) -> Result<Vec<f32>> {
    let values = read_f32(tensor)?;
    if values.iter().all(|x| x.is_finite()) {
        Ok(values)
    } else {
        Err(GpuError::NonFinite)
    }
}

/// Read an integer tensor back as `i64`.
pub fn read_i64<const D: usize>(tensor: Tensor<D, Int>) -> Result<Vec<i64>> {
    let data = tensor
        .try_into_data()
        .map_err(|err| GpuError::Readback(err.to_string()))?;
    data.try_into_vec_as::<i64>()
        .map_err(|err| GpuError::Readback(format!("{err:?}")))
}
