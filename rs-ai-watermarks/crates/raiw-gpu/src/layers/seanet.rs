//! SEANet encoder/decoder in the two parameter layouts the ported models ship:
//!
//! - [`SeanetLayout::Moshi`] (`audioseal.libs.moshi.modules.seanet`): `model.<i>.conv.conv.*`,
//!   identity skips, zero padding.
//! - [`SeanetLayout::Transformers`] (`EncodecModel`): `layers.<i>.conv.*`, 1x1 conv shortcuts,
//!   reflect padding, causal convolutions.

use burn::tensor::activation::elu;
use burn::tensor::ops::PadMode;
use burn::tensor::{Device, Tensor, s};

use super::conv::{Conv1d, Conv1dSpec, ConvTranspose1d, ConvTranspose1dSpec};
use super::lstm::Lstm;
use crate::error::Result;
use crate::weights::StateDict;

/// How `StreamingConv1d` fills its padding (`pad_mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadKind {
    Zeros,
    /// `pad1d(mode="reflect")`, zero-extending inputs shorter than the padding first.
    Reflect,
}

/// `StreamingConv1d`/`EncodecConv1d`: a convolution padded so every input frame is covered.
#[derive(Debug, Clone)]
pub struct StreamingConv1d {
    conv: Conv1d,
    causal: bool,
    pad: PadKind,
}

impl StreamingConv1d {
    /// Load `<conv_prefix>.{weight,bias}` (weight-norm folded).
    pub fn load(
        state: &mut StateDict,
        conv_prefix: &str,
        spec: Conv1dSpec,
        causal: bool,
        pad: PadKind,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            conv: Conv1d::load(state, conv_prefix, spec, device)?,
            causal,
            pad,
        })
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let spec = self.conv.spec();
        let kernel = spec.effective_kernel();
        let padding_total = kernel.saturating_sub(spec.stride);
        let extra = extra_padding(x.dims()[2], kernel, spec.stride, padding_total);
        let (left, right) = if self.causal {
            (padding_total, extra)
        } else {
            let right = padding_total / 2;
            (padding_total - right, right + extra)
        };
        match self.pad {
            PadKind::Zeros => self.conv.forward_padded(x, (left, right)),
            PadKind::Reflect => self.conv.forward_padded(reflect_pad(x, left, right), (0, 0)),
        }
    }
}

/// `StreamingConvTranspose1d`/`EncodecConvTranspose1d`: transposed conv trimmed back to `len * stride`.
#[derive(Debug, Clone)]
pub struct StreamingConvTranspose1d {
    conv: ConvTranspose1d,
    causal: bool,
}

impl StreamingConvTranspose1d {
    pub fn load(
        state: &mut StateDict,
        conv_prefix: &str,
        spec: ConvTranspose1dSpec,
        causal: bool,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            conv: ConvTranspose1d::load(state, conv_prefix, spec, device)?,
            causal,
        })
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let spec = self.conv.spec();
        let padding_total = spec.kernel.saturating_sub(spec.stride);
        let y = self.conv.forward(x);
        let right = if self.causal {
            padding_total
        } else {
            padding_total / 2
        };
        let left = padding_total - right;
        let end = y.dims()[2] - right;
        y.slice(s![.., .., left..end])
    }
}

/// Reflect-pad the time axis, zero-extending first when the input is too short to mirror.
pub fn reflect_pad(x: Tensor<3>, left: usize, right: usize) -> Tensor<3> {
    let length = x.dims()[2];
    let widest = left.max(right);
    let extra = if length <= widest { widest - length + 1 } else { 0 };
    let x = if extra > 0 {
        x.pad([(0, extra)], PadMode::Constant(0.0))
    } else {
        x
    };
    let padded = x.pad([(left, right)], PadMode::Reflect);
    if extra > 0 {
        let end = padded.dims()[2] - extra;
        padded.slice(s![.., .., 0..end])
    } else {
        padded
    }
}

/// Extra right padding so the last window is full (`get_extra_padding_for_conv1d`).
pub fn extra_padding(length: usize, kernel: usize, stride: usize, padding_total: usize) -> usize {
    let frames = (length as f64 - kernel as f64 + padding_total as f64) / stride as f64 + 1.0;
    let ideal = (frames.ceil() as i64 - 1) * stride as i64 + (kernel as i64 - padding_total as i64);
    (ideal - length as i64).max(0) as usize
}

/// Parameter naming and block conventions of a SEANet checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeanetLayout {
    Moshi,
    Transformers,
}

impl SeanetLayout {
    fn stage(self, prefix: &str, index: usize) -> String {
        match self {
            Self::Moshi => format!("{prefix}.model.{index}"),
            Self::Transformers => format!("{prefix}.layers.{index}"),
        }
    }

    fn conv(self, module: &str) -> String {
        match self {
            Self::Moshi => format!("{module}.conv.conv"),
            Self::Transformers => format!("{module}.conv"),
        }
    }

    fn conv_transpose(self, module: &str) -> String {
        match self {
            Self::Moshi => format!("{module}.convtr.convtr"),
            Self::Transformers => format!("{module}.conv"),
        }
    }
}

/// Hyper-parameters shared by SEANet encoders and decoders.
#[derive(Debug, Clone, PartialEq)]
pub struct SeanetConfig {
    pub channels: usize,
    pub dimension: usize,
    pub n_filters: usize,
    pub n_residual_layers: usize,
    /// Decoder-order ratios; the encoder walks them reversed.
    pub ratios: Vec<usize>,
    pub kernel_size: usize,
    pub last_kernel_size: usize,
    pub residual_kernel_size: usize,
    pub dilation_base: usize,
    pub compress: usize,
    pub lstm_layers: usize,
    pub causal: bool,
    pub elu_alpha: f64,
    pub pad: PadKind,
    /// Identity skip in resnet blocks; otherwise a 1x1 `shortcut` convolution.
    pub true_skip: bool,
    pub layout: SeanetLayout,
}

impl SeanetConfig {
    pub fn hop_length(&self) -> usize {
        self.ratios.iter().product()
    }

    fn conv(
        &self,
        state: &mut StateDict,
        module: &str,
        spec: Conv1dSpec,
        device: &Device,
    ) -> Result<StreamingConv1d> {
        StreamingConv1d::load(
            state,
            &self.layout.conv(module),
            spec,
            self.causal,
            self.pad,
            device,
        )
    }
}

/// `SEANetResnetBlock`: `skip(x) + conv1x1(elu(conv_k(elu(x))))`.
#[derive(Debug, Clone)]
pub struct SeanetResnetBlock {
    first: StreamingConv1d,
    second: StreamingConv1d,
    shortcut: Option<StreamingConv1d>,
    alpha: f64,
}

impl SeanetResnetBlock {
    pub fn load(
        state: &mut StateDict,
        prefix: &str,
        dim: usize,
        dilation: usize,
        config: &SeanetConfig,
        device: &Device,
    ) -> Result<Self> {
        let hidden = dim / config.compress;
        let first_spec = Conv1dSpec::new(dim, hidden, config.residual_kernel_size).dilation(dilation);
        let first = config.conv(state, &format!("{prefix}.block.1"), first_spec, device)?;
        let second = config.conv(
            state,
            &format!("{prefix}.block.3"),
            Conv1dSpec::new(hidden, dim, 1),
            device,
        )?;
        let shortcut = if config.true_skip {
            None
        } else {
            Some(config.conv(
                state,
                &format!("{prefix}.shortcut"),
                Conv1dSpec::new(dim, dim, 1),
                device,
            )?)
        };
        Ok(Self {
            first,
            second,
            shortcut,
            alpha: config.elu_alpha,
        })
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let y = self
            .second
            .forward(elu(self.first.forward(elu(x.clone(), self.alpha)), self.alpha));
        match &self.shortcut {
            Some(shortcut) => shortcut.forward(x) + y,
            None => x + y,
        }
    }
}

#[derive(Debug, Clone)]
enum Stage {
    Conv(StreamingConv1d),
    ConvTranspose(StreamingConvTranspose1d),
    Resnet(SeanetResnetBlock),
    Elu,
    /// LSTM over time with a residual skip, on `[batch, channels, time]`.
    Lstm(Lstm),
}

impl Stage {
    fn forward(&self, x: Tensor<3>, alpha: f64) -> Tensor<3> {
        match self {
            Self::Conv(conv) => conv.forward(x),
            Self::ConvTranspose(conv) => conv.forward(x),
            Self::Resnet(block) => block.forward(x),
            Self::Elu => elu(x, alpha),
            Self::Lstm(lstm) => {
                let time_major = x.swap_dims(1, 2);
                (lstm.forward(time_major.clone()) + time_major).swap_dims(1, 2)
            }
        }
    }
}

/// Stages addressed by their `nn.Sequential`/`ModuleList` index (parameter-free ones included).
struct StageBuilder<'a> {
    prefix: &'a str,
    config: &'a SeanetConfig,
    stages: Vec<Stage>,
}

impl<'a> StageBuilder<'a> {
    fn new(prefix: &'a str, config: &'a SeanetConfig) -> Self {
        Self {
            prefix,
            config,
            stages: Vec::new(),
        }
    }

    fn name(&self) -> String {
        self.config.layout.stage(self.prefix, self.stages.len())
    }

    fn conv(&mut self, state: &mut StateDict, spec: Conv1dSpec, device: &Device) -> Result<()> {
        let conv = self.config.conv(state, &self.name(), spec, device)?;
        self.stages.push(Stage::Conv(conv));
        Ok(())
    }

    fn conv_transpose(
        &mut self,
        state: &mut StateDict,
        spec: ConvTranspose1dSpec,
        device: &Device,
    ) -> Result<()> {
        let name = self.config.layout.conv_transpose(&self.name());
        let conv = StreamingConvTranspose1d::load(state, &name, spec, self.config.causal, device)?;
        self.stages.push(Stage::ConvTranspose(conv));
        Ok(())
    }

    fn resnets(&mut self, state: &mut StateDict, dim: usize, device: &Device) -> Result<()> {
        for j in 0..self.config.n_residual_layers {
            let dilation = self.config.dilation_base.pow(j as u32);
            let block = SeanetResnetBlock::load(state, &self.name(), dim, dilation, self.config, device)?;
            self.stages.push(Stage::Resnet(block));
        }
        Ok(())
    }

    fn lstm(&mut self, state: &mut StateDict, width: usize, device: &Device) -> Result<()> {
        if self.config.lstm_layers > 0 {
            let prefix = format!("{}.lstm", self.name());
            self.stages.push(Stage::Lstm(Lstm::load(
                state,
                &prefix,
                width,
                width,
                self.config.lstm_layers,
                device,
            )?));
        }
        Ok(())
    }

    fn elu(&mut self) {
        self.stages.push(Stage::Elu);
    }
}

/// `SEANetEncoder`: conv, (resnet blocks, ELU, strided conv) per ratio, optional LSTM, ELU, conv.
#[derive(Debug, Clone)]
pub struct SeanetEncoder {
    stages: Vec<Stage>,
    alpha: f64,
}

impl SeanetEncoder {
    pub fn load(state: &mut StateDict, prefix: &str, config: &SeanetConfig, device: &Device) -> Result<Self> {
        let mut builder = StageBuilder::new(prefix, config);
        builder.conv(
            state,
            Conv1dSpec::new(config.channels, config.n_filters, config.kernel_size),
            device,
        )?;
        let mut width = config.n_filters;
        for &ratio in config.ratios.iter().rev() {
            builder.resnets(state, width, device)?;
            builder.elu();
            builder.conv(
                state,
                Conv1dSpec::new(width, width * 2, ratio * 2).stride(ratio),
                device,
            )?;
            width *= 2;
        }
        builder.lstm(state, width, device)?;
        builder.elu();
        builder.conv(
            state,
            Conv1dSpec::new(width, config.dimension, config.last_kernel_size),
            device,
        )?;
        Ok(Self {
            stages: builder.stages,
            alpha: config.elu_alpha,
        })
    }

    /// `[batch, channels, samples]` to `[batch, dimension, frames]`.
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.stages
            .iter()
            .fold(x, |x, stage| stage.forward(x, self.alpha))
    }
}

/// `SEANetDecoder`: conv, optional LSTM, (ELU, transposed conv, resnet blocks) per ratio, ELU, conv.
#[derive(Debug, Clone)]
pub struct SeanetDecoder {
    stages: Vec<Stage>,
    alpha: f64,
}

impl SeanetDecoder {
    pub fn load(state: &mut StateDict, prefix: &str, config: &SeanetConfig, device: &Device) -> Result<Self> {
        let mut builder = StageBuilder::new(prefix, config);
        let mut width = config.n_filters << config.ratios.len();
        builder.conv(
            state,
            Conv1dSpec::new(config.dimension, width, config.kernel_size),
            device,
        )?;
        builder.lstm(state, width, device)?;
        for &ratio in &config.ratios {
            builder.elu();
            builder.conv_transpose(
                state,
                ConvTranspose1dSpec::new(width, width / 2, ratio * 2, ratio),
                device,
            )?;
            width /= 2;
            builder.resnets(state, width, device)?;
        }
        builder.elu();
        builder.conv(
            state,
            Conv1dSpec::new(width, config.channels, config.last_kernel_size),
            device,
        )?;
        Ok(Self {
            stages: builder.stages,
            alpha: config.elu_alpha,
        })
    }

    /// `[batch, dimension, frames]` to `[batch, channels, frames * hop]`.
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.stages
            .iter()
            .fold(x, |x, stage| stage.forward(x, self.alpha))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extra_padding_matches_moshi() {
        assert_eq!(extra_padding(5, 4, 2, 2), 1);
        assert_eq!(extra_padding(16000, 4, 2, 2), 0);
        assert_eq!(extra_padding(16001, 16, 8, 8), 7);
        assert_eq!(extra_padding(1, 7, 1, 6), 0);
    }

    #[test]
    fn reflect_pad_mirrors_and_handles_short_inputs() {
        let device = Device::flex();
        let x = Tensor::<3>::from_floats([[[1.0, 2.0, 3.0, 4.0]]], &device);
        let padded = crate::io::read_f32(reflect_pad(x, 2, 1)).unwrap();
        assert_eq!(padded, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0]);
        let short = Tensor::<3>::from_floats([[[1.0, 2.0]]], &device);
        let padded = crate::io::read_f32(reflect_pad(short, 3, 0)).unwrap();
        assert_eq!(padded, vec![0.0, 0.0, 2.0, 1.0, 2.0]);
    }
}
