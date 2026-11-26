//! Multi-layer unidirectional LSTM with `torch.nn.LSTM` weights, batch-first.
//!
//! The input projection is one device matmul; the recurrence runs on the host instead of
//! a chain of tiny device dispatches per time step.

use burn::tensor::{Device, Tensor, TensorData};

use crate::error::Result;
use crate::io::read_f32;
use crate::weights::StateDict;

#[derive(Debug, Clone)]
struct LstmLayer {
    /// `W_ih^T`, shape `[1, input, 4 * hidden]`.
    w_ih: Tensor<3>,
    /// `b_ih + b_hh`, shape `[1, 1, 4 * hidden]`.
    bias: Tensor<3>,
    /// Host `W_hh^T`, row-major `[hidden, 4 * hidden]`.
    w_hh: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Lstm {
    layers: Vec<LstmLayer>,
    hidden: usize,
}

impl Lstm {
    /// Load `<prefix>.{weight_ih,weight_hh,bias_ih,bias_hh}_l<k>` for every layer (gate order i, f, g, o).
    pub fn load(
        state: &mut StateDict,
        prefix: &str,
        input: usize,
        hidden: usize,
        num_layers: usize,
        device: &Device,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let width = if layer == 0 { input } else { hidden };
            let w_ih: Tensor<2> = state.tensor(
                &format!("{prefix}.weight_ih_l{layer}"),
                [4 * hidden, width],
                device,
            )?;
            let w_hh = state.take(&format!("{prefix}.weight_hh_l{layer}"), &[4 * hidden, hidden])?;
            let b_ih: Tensor<1> =
                state.tensor(&format!("{prefix}.bias_ih_l{layer}"), [4 * hidden], device)?;
            let b_hh: Tensor<1> =
                state.tensor(&format!("{prefix}.bias_hh_l{layer}"), [4 * hidden], device)?;
            let mut w_hh_t = vec![0.0; 4 * hidden * hidden];
            for (gate, row) in w_hh.data.chunks_exact(hidden).enumerate() {
                for (k, &w) in row.iter().enumerate() {
                    w_hh_t[k * 4 * hidden + gate] = w;
                }
            }
            layers.push(LstmLayer {
                w_ih: w_ih.transpose().unsqueeze_dim(0),
                bias: (b_ih + b_hh).reshape([1, 1, 4 * hidden]),
                w_hh: w_hh_t,
            });
        }
        Ok(Self { layers, hidden })
    }

    /// Run `[batch, time, input]` from zero state and return `[batch, time, hidden]`.
    pub fn forward(&self, x: Tensor<3>) -> Result<Tensor<3>> {
        self.layers
            .iter()
            .try_fold(x, |x, layer| self.forward_layer(layer, x))
    }

    fn forward_layer(&self, layer: &LstmLayer, x: Tensor<3>) -> Result<Tensor<3>> {
        let [batch, steps, _] = x.dims();
        let (hidden, width) = (self.hidden, 4 * self.hidden);
        let device = x.device();
        let projected = read_f32(x.matmul(layer.w_ih.clone()) + layer.bias.clone())?;
        let mut out = vec![0.0f32; batch * steps * hidden];
        let mut gates = vec![0.0f32; width];
        for (inputs, outputs) in projected
            .chunks_exact(steps * width)
            .zip(out.chunks_exact_mut(steps * hidden))
        {
            let mut h = vec![0.0f32; hidden];
            let mut c = vec![0.0f32; hidden];
            for (input, output) in inputs.chunks_exact(width).zip(outputs.chunks_exact_mut(hidden)) {
                gates.copy_from_slice(input);
                for (&h_k, row) in h.iter().zip(layer.w_hh.chunks_exact(width)) {
                    for (gate, &w) in gates.iter_mut().zip(row) {
                        *gate += h_k * w;
                    }
                }
                let (i, rest) = gates.split_at(hidden);
                let (f, rest) = rest.split_at(hidden);
                let (g, o) = rest.split_at(hidden);
                for k in 0..hidden {
                    c[k] = sigmoid(f[k]) * c[k] + sigmoid(i[k]) * g[k].tanh();
                    h[k] = sigmoid(o[k]) * c[k].tanh();
                }
                output.copy_from_slice(&h);
            }
        }
        Ok(Tensor::from_data(
            TensorData::new(out, [batch, steps, hidden]),
            &device,
        ))
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use burn::tensor::activation;
    use burn::tensor::s;

    use super::*;
    use crate::weights::HostTensor;

    fn seeded(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (f64::from(state) / 4_294_967_296.0 - 0.5) as f32
            })
            .collect()
    }

    /// The device-side recurrence as burn ops, gate order i, f, g, o.
    fn reference(state: &StateDict, x: Tensor<3>, hidden: usize, layers: usize) -> Vec<f32> {
        let device = x.device();
        let mut x = x;
        for layer in 0..layers {
            let get = |name: &str| state_tensor(state, &format!("lstm.{name}_l{layer}"), &device);
            let (w_ih, w_hh) = (get("weight_ih").transpose(), get("weight_hh").transpose());
            let bias = (get("bias_ih") + get("bias_hh")).unsqueeze_dim::<3>(0);
            let [batch, steps, _] = x.dims();
            let projected = x.matmul(w_ih.unsqueeze_dim(0)) + bias;
            let mut h = Tensor::<2>::zeros([batch, hidden], &device);
            let mut c = Tensor::<2>::zeros([batch, hidden], &device);
            let mut outputs = Vec::with_capacity(steps);
            for t in 0..steps {
                let gates = projected
                    .clone()
                    .slice(s![.., t, ..])
                    .reshape([batch, 4 * hidden])
                    + h.matmul(w_hh.clone());
                let gate = |k: usize| gates.clone().narrow(1, k * hidden, hidden);
                c = activation::sigmoid(gate(1)) * c
                    + activation::sigmoid(gate(0)) * activation::tanh(gate(2));
                h = activation::sigmoid(gate(3)) * activation::tanh(c.clone());
                outputs.push(h.clone().unsqueeze_dim::<3>(1));
            }
            x = Tensor::cat(outputs, 1);
        }
        read_f32(x).unwrap()
    }

    fn state_tensor(state: &StateDict, name: &str, device: &Device) -> Tensor<2> {
        let mut copy = state.clone();
        let host = copy.take_any(name).unwrap();
        let shape = match host.shape[..] {
            [rows, cols] => [rows, cols],
            [len] => [1, len],
            _ => unreachable!(),
        };
        Tensor::from_data(TensorData::new(host.data, shape), device)
    }

    #[test]
    fn host_recurrence_matches_device_ops() {
        let (input, hidden, layers, batch, steps) = (6, 8, 2, 2, 23);
        let mut tensors = std::collections::BTreeMap::new();
        for layer in 0..layers {
            let width = if layer == 0 { input } else { hidden };
            let shapes = [
                ("weight_ih", vec![4 * hidden, width]),
                ("weight_hh", vec![4 * hidden, hidden]),
                ("bias_ih", vec![4 * hidden]),
                ("bias_hh", vec![4 * hidden]),
            ];
            for (i, (name, shape)) in shapes.into_iter().enumerate() {
                let data = seeded(shape.iter().product(), (layer * 4 + i) as u32);
                tensors.insert(
                    format!("lstm.{name}_l{layer}"),
                    HostTensor::new(shape, data).unwrap(),
                );
            }
        }
        let state = StateDict::from_tensors("seeded", tensors);
        let device = Device::flex();
        let x: Tensor<3> = Tensor::from_data(
            TensorData::new(seeded(batch * steps * input, 99), [batch, steps, input]),
            &device,
        );
        let expected = reference(&state, x.clone(), hidden, layers);
        let lstm = Lstm::load(&mut state.clone(), "lstm", input, hidden, layers, &device).unwrap();
        let actual = read_f32(lstm.forward(x).unwrap()).unwrap();
        let diff = expected
            .iter()
            .zip(&actual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(diff < 1e-5, "max diff {diff}");
    }
}
