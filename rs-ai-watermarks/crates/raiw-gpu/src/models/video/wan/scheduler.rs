//! `UniPCMultistepScheduler` as pinned by `scheduler/scheduler_config.json` plus
//! `flow_shift=3.0`: flow sigmas, `flow_prediction`, `predict_x0`, `bh2`, order 2,
//! `lower_order_final`, final sigma zero. State lives on host `f32` buffers; the scalar
//! coefficients follow torch's 0-dim `float32` arithmetic step by step.

use crate::error::{GpuError, Result};

const TRAIN_TIMESTEPS: usize = 1000;
const SOLVER_ORDER: usize = 2;

#[derive(Debug, Clone)]
pub struct UniPcScheduler {
    sigmas: Vec<f32>,
    timesteps: Vec<i64>,
    step_index: Option<usize>,
    /// Converted (`x0`) model outputs, oldest first.
    model_outputs: [Option<Vec<f32>>; SOLVER_ORDER],
    lower_order_nums: usize,
    last_sample: Option<Vec<f32>>,
    this_order: usize,
}

impl UniPcScheduler {
    /// `set_timesteps(steps)` with `use_flow_sigmas` and the given `flow_shift`.
    pub fn new(steps: usize, flow_shift: f64) -> Result<Self> {
        if steps == 0 {
            return Err(GpuError::Input("UniPC needs at least one step".to_owned()));
        }
        let stop = 1.0 / TRAIN_TIMESTEPS as f64;
        let delta = (stop - 1.0) / steps as f64;
        let mut sigmas: Vec<f64> = (0..steps)
            .map(|i| {
                let s = i as f64 * delta + 1.0;
                flow_shift * s / (1.0 + (flow_shift - 1.0) * s)
            })
            .collect();
        if (sigmas[0] - 1.0).abs() < 1e-6 {
            sigmas[0] -= 1e-6;
        }
        let timesteps = sigmas
            .iter()
            .map(|s| (s * TRAIN_TIMESTEPS as f64) as i64)
            .collect();
        let sigmas = sigmas
            .iter()
            .map(|&s| s as f32)
            .chain(std::iter::once(0.0))
            .collect();
        Ok(Self {
            sigmas,
            timesteps,
            step_index: None,
            model_outputs: [None, None],
            lower_order_nums: 0,
            last_sample: None,
            this_order: 1,
        })
    }

    pub fn timesteps(&self) -> &[i64] {
        &self.timesteps
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    /// `WanVideoToVideoPipeline.get_timesteps`: the schedule tail that `strength` keeps.
    pub fn strength_timesteps(&self, strength: f64) -> &[i64] {
        let steps = self.timesteps.len();
        let init = ((steps as f64 * strength) as usize).min(steps);
        &self.timesteps[steps - init..]
    }

    /// `index_for_timestep`: the second match when a timestep repeats, the last index if absent.
    fn index_for(&self, timestep: i64) -> usize {
        let matches: Vec<usize> = (0..self.timesteps.len())
            .filter(|&i| self.timesteps[i] == timestep)
            .collect();
        match matches.as_slice() {
            [] => self.timesteps.len() - 1,
            [only] => *only,
            [_, second, ..] => *second,
        }
    }

    /// `add_noise`: `(1 - sigma) * original + sigma * noise` at `timestep`.
    pub fn add_noise(&self, original: &[f32], noise: &[f32], timestep: i64) -> Result<Vec<f32>> {
        check_len(noise.len(), original.len(), "noise")?;
        let sigma = self.sigmas[self.index_for(timestep)];
        let alpha = 1.0 - sigma;
        Ok(original
            .iter()
            .zip(noise)
            .map(|(&x, &n)| alpha * x + sigma * n)
            .collect())
    }

    /// One `step(model_output, timestep, sample)`: UniC correction, then the UniP prediction.
    pub fn step(&mut self, model_output: &[f32], timestep: i64, sample: &[f32]) -> Result<Vec<f32>> {
        check_len(model_output.len(), sample.len(), "model output")?;
        let index = self.step_index.unwrap_or_else(|| self.index_for(timestep));
        if index + 1 >= self.sigmas.len() {
            return Err(GpuError::Input(format!(
                "UniPC stepped past its {} steps",
                self.timesteps.len()
            )));
        }
        let sigma = self.sigmas[index];
        let converted: Vec<f32> = sample
            .iter()
            .zip(model_output)
            .map(|(&x, &v)| x - sigma * v)
            .collect();
        let sample = match self.last_sample.take() {
            Some(last) if index > 0 => self.corrector(&converted, &last, index),
            _ => sample.to_vec(),
        };
        self.model_outputs.rotate_left(1);
        self.model_outputs[SOLVER_ORDER - 1] = Some(converted);
        let order = SOLVER_ORDER.min(self.timesteps.len() - index);
        self.this_order = order.min(self.lower_order_nums + 1);
        let prev = self.predictor(&sample, index);
        self.last_sample = Some(sample);
        if self.lower_order_nums < SOLVER_ORDER {
            self.lower_order_nums += 1;
        }
        self.step_index = Some(index + 1);
        Ok(prev)
    }

    /// `(alpha, sigma, lambda)` of schedule entry `i`.
    fn point(&self, i: usize) -> (f32, f32, f32) {
        let sigma = self.sigmas[i];
        let alpha = 1.0 - sigma;
        (alpha, sigma, alpha.ln() - sigma.ln())
    }

    /// `multistep_uni_p_bh_update` (predict_x0, bh2).
    fn predictor(&self, x: &[f32], index: usize) -> Vec<f32> {
        let m0 = self.output(1);
        let (alpha_t, sigma_t, lambda_t) = self.point(index + 1);
        let (_, sigma_s0, lambda_s0) = self.point(index);
        let h = lambda_t - lambda_s0;
        let hh = -h;
        let h_phi_1 = hh.exp_m1();
        let b_h = hh.exp_m1();
        let base = |x: f32, m0: f32| sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0;
        if self.this_order < 2 {
            return x.iter().zip(m0).map(|(&x, &m)| base(x, m)).collect();
        }
        let m1 = self.output(2);
        let (_, _, lambda_s1) = self.point(index - 1);
        let rk = (lambda_s1 - lambda_s0) / h;
        let rho = 0.5f32;
        let coeff = alpha_t * b_h;
        x.iter()
            .zip(m0)
            .zip(m1)
            .map(|((&x, &m0), &m1)| base(x, m0) - coeff * (rho * ((m1 - m0) / rk)))
            .collect()
    }

    /// `multistep_uni_c_bh_update` (predict_x0, bh2) with the previous step's order.
    fn corrector(&self, model_t: &[f32], last: &[f32], index: usize) -> Vec<f32> {
        let m0 = self.output(1);
        let (alpha_t, sigma_t, lambda_t) = self.point(index);
        let (_, sigma_s0, lambda_s0) = self.point(index - 1);
        let h = lambda_t - lambda_s0;
        let hh = -h;
        let h_phi_1 = hh.exp_m1();
        let b_h = hh.exp_m1();
        let base = |x: f32, m0: f32| sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0;
        let coeff = alpha_t * b_h;
        // b_i = h_phi_k * i! / B_h with h_phi_k = h_phi_{k-1} / hh - 1 / (k)!
        let h_phi_k1 = h_phi_1 / hh - 1.0;
        let b1 = h_phi_k1 / b_h;
        if self.this_order < 2 || index < 2 {
            let rho = 0.5f32;
            return last
                .iter()
                .zip(m0)
                .zip(model_t)
                .map(|((&x, &m0), &mt)| base(x, m0) - coeff * (rho * (mt - m0)))
                .collect();
        }
        let h_phi_k2 = h_phi_k1 / hh - 1.0 / 2.0;
        let b2 = h_phi_k2 * 2.0 / b_h;
        let m1 = self.output(2);
        let (_, _, lambda_s1) = self.point(index - 2);
        let rk = (lambda_s1 - lambda_s0) / h;
        // torch.linalg.solve([[1, 1], [rk, 1]], [b1, b2])
        let rho0 = (b1 - b2) / (1.0 - rk);
        let rho1 = b1 - rho0;
        last.iter()
            .zip(m0)
            .zip(m1)
            .zip(model_t)
            .map(|(((&x, &m0), &m1), &mt)| {
                let corr = rho0 * ((m1 - m0) / rk);
                base(x, m0) - coeff * (corr + rho1 * (mt - m0))
            })
            .collect()
    }

    /// The `back`-th most recent converted output (1 = newest).
    fn output(&self, back: usize) -> &[f32] {
        self.model_outputs[SOLVER_ORDER - back]
            .as_deref()
            .unwrap_or_default()
    }
}

fn check_len(actual: usize, expected: usize, what: &str) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(GpuError::Input(format!(
            "UniPC {what} holds {actual} values, expected {expected}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flow_schedule_starts_below_one_and_ends_at_zero() {
        let scheduler = UniPcScheduler::new(20, 3.0).unwrap();
        assert_eq!(scheduler.timesteps()[0], 999);
        assert_eq!(scheduler.sigmas().len(), 21);
        assert_eq!(scheduler.sigmas()[20], 0.0);
        assert_eq!(scheduler.strength_timesteps(0.38).len(), 7);
        assert_eq!(scheduler.strength_timesteps(0.12).len(), 2);
    }
}
