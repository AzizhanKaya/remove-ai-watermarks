//! Multi-layer unidirectional LSTM with `torch.nn.LSTM` weights, batch-first.

use burn::tensor::activation::{sigmoid, tanh};
use burn::tensor::{Device, Tensor, s};

use crate::error::Result;
use crate::weights::StateDict;

#[derive(Debug, Clone)]
struct LstmLayer {
    /// `W_ih^T`, shape `[input, 4 * hidden]`.
    w_ih: Tensor<2>,
    /// `W_hh^T`, shape `[hidden, 4 * hidden]`.
    w_hh: Tensor<2>,
    /// `b_ih + b_hh`, shape `[1, 1, 4 * hidden]`.
    bias: Tensor<3>,
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
            let w_hh: Tensor<2> = state.tensor(
                &format!("{prefix}.weight_hh_l{layer}"),
                [4 * hidden, hidden],
                device,
            )?;
            let b_ih: Tensor<1> =
                state.tensor(&format!("{prefix}.bias_ih_l{layer}"), [4 * hidden], device)?;
            let b_hh: Tensor<1> =
                state.tensor(&format!("{prefix}.bias_hh_l{layer}"), [4 * hidden], device)?;
            layers.push(LstmLayer {
                w_ih: w_ih.transpose(),
                w_hh: w_hh.transpose(),
                bias: (b_ih + b_hh).reshape([1, 1, 4 * hidden]),
            });
        }
        Ok(Self { layers, hidden })
    }

    /// Run `[batch, time, input]` from zero state and return `[batch, time, hidden]`.
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.layers
            .iter()
            .fold(x, |x, layer| self.forward_layer(layer, x))
    }

    fn forward_layer(&self, layer: &LstmLayer, x: Tensor<3>) -> Tensor<3> {
        let [batch, steps, _] = x.dims();
        let hidden = self.hidden;
        let device = x.device();
        let projected = x.matmul(layer.w_ih.clone().unsqueeze_dim(0)) + layer.bias.clone();
        let mut h = Tensor::<2>::zeros([batch, hidden], &device);
        let mut c = Tensor::<2>::zeros([batch, hidden], &device);
        let mut outputs = Vec::with_capacity(steps);
        for t in 0..steps {
            let gates = projected
                .clone()
                .slice(s![.., t, ..])
                .reshape([batch, 4 * hidden])
                + h.matmul(layer.w_hh.clone());
            let gate = |k: usize| gates.clone().narrow(1, k * hidden, hidden);
            c = sigmoid(gate(1)) * c + sigmoid(gate(0)) * tanh(gate(2));
            h = sigmoid(gate(3)) * tanh(c.clone());
            outputs.push(h.clone().unsqueeze_dim::<3>(1));
        }
        Tensor::cat(outputs, 1)
    }
}
