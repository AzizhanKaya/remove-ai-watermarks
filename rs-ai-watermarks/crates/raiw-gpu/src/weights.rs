//! Checkpoint loading through burn-store, weight-norm folding, and typed tensor extraction.
//!
//! A [`StateDict`] is read once on the host (PyTorch `.pth`/`.bin` via `PytorchStore`,
//! `.safetensors` via `SafetensorsStore`, both with regex key remapping), weight-normed
//! pairs are folded into plain `weight` tensors, and model builders then take each tensor
//! by name with a shape check, so a mismatched checkpoint fails loudly instead of silently.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use burn::tensor::{Device, Int, Tensor, TensorData};
use burn_store::{KeyRemapper, ModuleStore, PytorchStore, SafetensorsStore};

use crate::error::{GpuError, Result};

/// A host-resident `f32` tensor taken from a checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct HostTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl HostTensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        if shape.iter().product::<usize>() != data.len() {
            return Err(GpuError::Config(format!(
                "tensor data length {} does not match shape {shape:?}",
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }

    pub fn to_device<const D: usize>(&self, device: &Device) -> Tensor<D> {
        Tensor::from_data(TensorData::new(self.data.clone(), self.shape.clone()), device)
    }
}

/// Every tensor of one checkpoint, keyed by (remapped) PyTorch name.
#[derive(Debug, Clone)]
pub struct StateDict {
    source: PathBuf,
    tensors: BTreeMap<String, HostTensor>,
}

/// Regex remapping applied by burn-store while reading, as `(pattern, replacement)` pairs.
pub type Remap<'a> = &'a [(&'a str, &'a str)];

impl StateDict {
    /// Read a PyTorch pickle checkpoint, optionally descending into `top_level_key` (e.g. `"model"`).
    ///
    /// burn-store's `PytorchStore` reads it first; checkpoints whose pickles embed objects it
    /// rejects (omegaconf configs keyed by tuples) fall back to the tolerant zip reader.
    pub fn from_pytorch(path: &Path, top_level_key: Option<&str>, remap: Remap<'_>) -> Result<Self> {
        // Keep PyTorch's own `nn.Sequential` indices; builders address them verbatim
        let mut store = PytorchStore::from_file(path)
            .map_indices_contiguous(false)
            .remap(remapper(path, remap)?);
        if let Some(key) = top_level_key {
            store = store.with_top_level_key(key);
        }
        match Self::from_store(path, &mut store) {
            Ok(state) => Ok(state),
            Err(primary) => {
                let tensors =
                    crate::torch_pickle::read_checkpoint(path, top_level_key).map_err(|fallback| {
                        GpuError::Weights {
                            path: path.to_path_buf(),
                            reason: format!("{primary}; fallback reader: {fallback}"),
                        }
                    })?;
                tracing::debug!(path = %path.display(), "read checkpoint with the tolerant pickle reader");
                Ok(Self {
                    source: path.to_path_buf(),
                    tensors: remap_keys(path, tensors, remap)?,
                })
            }
        }
    }

    pub fn from_safetensors(path: &Path, remap: Remap<'_>) -> Result<Self> {
        let mut store = SafetensorsStore::from_file(path).remap(remapper(path, remap)?);
        Self::from_store(path, &mut store)
    }

    /// Pick the reader from the file extension.
    pub fn open(path: &Path, top_level_key: Option<&str>, remap: Remap<'_>) -> Result<Self> {
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("safetensors") => Self::from_safetensors(path, remap),
            _ => Self::from_pytorch(path, top_level_key, remap),
        }
    }

    fn from_store<S: ModuleStore>(path: &Path, store: &mut S) -> Result<Self> {
        let weights_error = |reason: String| GpuError::Weights {
            path: path.to_path_buf(),
            reason,
        };
        let packed = store
            .get_all_tensors()
            .map_err(|err| weights_error(err.to_string()))?;
        let mut tensors = BTreeMap::new();
        for (name, tensor) in packed {
            let data =
                burn_store::bridge::to_data(tensor).map_err(|err| weights_error(format!("{name}: {err}")))?;
            let shape = data.shape().to_vec();
            let values = data
                .try_into_vec_as::<f32>()
                .map_err(|err| weights_error(format!("{name}: {err:?}")))?;
            tensors.insert(name.clone(), HostTensor { shape, data: values });
        }
        Ok(Self {
            source: path.to_path_buf(),
            tensors,
        })
    }

    pub fn from_tensors(source: impl Into<PathBuf>, tensors: BTreeMap<String, HostTensor>) -> Self {
        Self {
            source: source.into(),
            tensors,
        }
    }

    pub fn source(&self) -> &Path {
        &self.source
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Replace every weight-normed `(g, v)` pair with its folded `weight = g * v / ||v||`.
    ///
    /// Handles both the legacy `weight_g`/`weight_v` hooks and the
    /// `parametrizations.weight.original0/1` form; the norm runs over every dim but 0.
    pub fn fold_weight_norm(&mut self) -> Result<usize> {
        const FORMS: [(&str, &str); 2] = [
            (".weight_g", ".weight_v"),
            (
                ".parametrizations.weight.original0",
                ".parametrizations.weight.original1",
            ),
        ];
        let mut folded = 0;
        for (g_suffix, v_suffix) in FORMS {
            let prefixes: Vec<String> = self
                .tensors
                .keys()
                .filter_map(|key| key.strip_suffix(g_suffix))
                .map(str::to_owned)
                .collect();
            for prefix in prefixes {
                let g = self.take_any(&format!("{prefix}{g_suffix}"))?;
                let v = self.take_any(&format!("{prefix}{v_suffix}"))?;
                self.tensors
                    .insert(format!("{prefix}.weight"), fold_pair(&prefix, &g, &v)?);
                folded += 1;
            }
        }
        Ok(folded)
    }

    /// Remove a tensor regardless of shape.
    pub fn take_any(&mut self, name: &str) -> Result<HostTensor> {
        self.tensors
            .remove(name)
            .ok_or_else(|| GpuError::MissingTensor(name.to_owned()))
    }

    /// Remove a tensor, checking its shape.
    pub fn take(&mut self, name: &str, shape: &[usize]) -> Result<HostTensor> {
        let tensor = self.take_any(name)?;
        if tensor.shape != shape {
            return Err(GpuError::ShapeMismatch {
                name: name.to_owned(),
                actual: tensor.shape,
                expected: shape.to_vec(),
            });
        }
        Ok(tensor)
    }

    /// Remove a tensor and upload it with a statically known rank.
    pub fn tensor<const D: usize>(
        &mut self,
        name: &str,
        shape: [usize; D],
        device: &Device,
    ) -> Result<Tensor<D>> {
        Ok(self.take(name, &shape)?.to_device(device))
    }

    /// Remove an optional tensor (e.g. a bias) and upload it.
    pub fn optional<const D: usize>(
        &mut self,
        name: &str,
        shape: [usize; D],
        device: &Device,
    ) -> Result<Option<Tensor<D>>> {
        if self.contains(name) {
            self.tensor(name, shape, device).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Drop tensors the inference graph never reads (buffers, unused codebooks).
    pub fn discard(&mut self, predicate: impl Fn(&str) -> bool) {
        self.tensors.retain(|name, _| !predicate(name));
    }

    /// Fail if any tensor was left unread, which signals an architecture mismatch.
    pub fn finish(self) -> Result<()> {
        if self.tensors.is_empty() {
            Ok(())
        } else {
            Err(GpuError::UnusedTensors(self.tensors.into_keys().collect()))
        }
    }
}

fn remapper(path: &Path, remap: Remap<'_>) -> Result<KeyRemapper> {
    KeyRemapper::from_patterns(remap.to_vec()).map_err(|err| GpuError::Weights {
        path: path.to_path_buf(),
        reason: format!("bad key remap: {err}"),
    })
}

/// Apply the same regex remapping as [`KeyRemapper`] to an already-read tensor map.
fn remap_keys(
    path: &Path,
    tensors: BTreeMap<String, HostTensor>,
    remap: Remap<'_>,
) -> Result<BTreeMap<String, HostTensor>> {
    let patterns = remapper(path, remap)?.to_regex_pairs();
    Ok(tensors
        .into_iter()
        .map(|(name, tensor)| {
            let renamed = patterns.iter().fold(name, |name, (regex, to)| {
                regex.replace_all(&name, to.as_str()).into_owned()
            });
            (renamed, tensor)
        })
        .collect())
}

fn fold_pair(prefix: &str, g: &HostTensor, v: &HostTensor) -> Result<HostTensor> {
    let rows = *v
        .shape
        .first()
        .ok_or_else(|| GpuError::Config(format!("{prefix}: scalar weight_v")))?;
    if g.data.len() != rows || rows == 0 || v.data.len() % rows != 0 {
        return Err(GpuError::ShapeMismatch {
            name: format!("{prefix}.weight_g"),
            actual: g.shape.clone(),
            expected: vec![rows],
        });
    }
    let row_len = v.data.len() / rows;
    let mut data = Vec::with_capacity(v.data.len());
    for (row, &gain) in v.data.chunks_exact(row_len).zip(&g.data) {
        let norm = row
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum::<f64>()
            .sqrt();
        let scale = f64::from(gain) / norm;
        data.extend(row.iter().map(|&x| (f64::from(x) * scale) as f32));
    }
    HostTensor::new(v.shape.clone(), data)
}

/// Upload integer codes (e.g. codebook indices) to a device.
pub fn int_tensor<const D: usize>(values: Vec<i64>, shape: [usize; D], device: &Device) -> Tensor<D, Int> {
    Tensor::from_data(TensorData::new(values, shape), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host(shape: &[usize], data: &[f32]) -> HostTensor {
        HostTensor::new(shape.to_vec(), data.to_vec()).unwrap()
    }

    #[test]
    fn folds_both_weight_norm_forms() {
        let mut tensors = BTreeMap::new();
        tensors.insert("a.weight_g".to_owned(), host(&[2, 1, 1], &[2.0, 1.0]));
        tensors.insert("a.weight_v".to_owned(), host(&[2, 1, 2], &[3.0, 4.0, 0.0, 5.0]));
        tensors.insert(
            "b.parametrizations.weight.original0".to_owned(),
            host(&[1, 1, 1], &[10.0]),
        );
        tensors.insert(
            "b.parametrizations.weight.original1".to_owned(),
            host(&[1, 2, 1], &[0.6, 0.8]),
        );
        tensors.insert("b.bias".to_owned(), host(&[1], &[0.5]));
        let mut state = StateDict::from_tensors("mem", tensors);
        assert_eq!(state.fold_weight_norm().unwrap(), 2);
        let a = state.take("a.weight", &[2, 1, 2]).unwrap();
        assert!(
            a.data
                .iter()
                .zip([1.2, 1.6, 0.0, 1.0])
                .all(|(x, y)| (x - y).abs() < 1e-6)
        );
        let b = state.take("b.weight", &[1, 2, 1]).unwrap();
        assert!(b.data.iter().zip([6.0, 8.0]).all(|(x, y)| (x - y).abs() < 1e-5));
        assert!(matches!(
            state.take("b.bias", &[2]),
            Err(GpuError::ShapeMismatch { .. })
        ));
        state.finish().unwrap();
    }

    #[test]
    fn unread_tensors_fail_finish() {
        let mut tensors = BTreeMap::new();
        tensors.insert("left.over".to_owned(), host(&[1], &[0.0]));
        let state = StateDict::from_tensors("mem", tensors);
        assert!(matches!(state.finish(), Err(GpuError::UnusedTensors(keys)) if keys == ["left.over"]));
    }
}
