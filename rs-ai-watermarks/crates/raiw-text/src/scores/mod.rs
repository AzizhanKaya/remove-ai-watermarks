//! Keyed local watermark scorers: the open Threefry SynthID reference and BLAKE2 simulations.
//!
//! The BLAKE2 scorers are research simulations, not vendor detectors; knowing a provider's key
//! does not make their hashing compatible. None of these verifies a production watermark.

pub mod threefry;

use std::collections::HashSet;

use blake2::Blake2sVar;
use blake2::digest::{Update, VariableOutput};
use serde::Serialize;

use crate::error::TextError;

/// One-sided `z` at `alpha = 0.01`.
pub const DEFAULT_ALPHA_001_Z: f64 = 2.326_347_874_040_840_8;
const TEXTGRAIN_SUB_PART: i64 = 0x1F3D_5B79;
const TEXTGRAIN_SUB_COST: i64 = 0x2C4E_6A8F;
const TEXTGRAIN_SUB_COL: i64 = 0x3B5D_7F91;
const TEXTSEAL_DUAL_KEY_OFFSET: i64 = 12345;

/// The simulated scheme a score belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum WatermarkScheme {
    Synthid,
    Gumbel,
    Textgrain,
    Textseal2,
    Kgw,
    Unigram,
}

/// Keyed statistical detector result for a token sequence.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct TextWatermarkScore {
    pub scheme: WatermarkScheme,
    pub z_score: f64,
    pub p_value: f64,
    pub detected: bool,
    pub num_tokens: usize,
    pub threshold_z: f64,
}

/// Frequentist score for the open single-key Threefry SynthID reference.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct SynthIDThreefryScore {
    pub p_value: f64,
    pub detected: bool,
    pub num_tokens: u64,
    pub num_ones: u64,
    pub num_trials: u64,
    pub alpha: f64,
}

fn one_sided_p_value(z_score: f64) -> f64 {
    0.5 * libm::erfc(z_score / std::f64::consts::SQRT_2)
}

fn make_score(
    scheme: WatermarkScheme,
    z_score: f64,
    num_tokens: usize,
    threshold_z: f64,
) -> TextWatermarkScore {
    if num_tokens == 0 {
        return TextWatermarkScore {
            scheme,
            z_score: 0.0,
            p_value: 0.5,
            detected: false,
            num_tokens: 0,
            threshold_z,
        };
    }
    TextWatermarkScore {
        scheme,
        z_score,
        p_value: one_sided_p_value(z_score),
        detected: z_score >= threshold_z,
        num_tokens,
        threshold_z,
    }
}

/// 64-bit keyed BLAKE2s digest of `(context..., token, key)` as signed little-endian words.
fn prf(context: &[i64], token_id: i64, key: i64) -> u64 {
    let mut hasher = Blake2sVar::new(8).unwrap_or_else(|_| unreachable!("8 is a valid BLAKE2s output size"));
    for item in context.iter().chain([&token_id, &key]) {
        hasher.update(&item.to_le_bytes());
    }
    let mut digest = [0u8; 8];
    let _ = hasher.finalize_variable(&mut digest);
    u64::from_le_bytes(digest)
}

fn prf_uniform(context: &[i64], token_id: i64, key: i64) -> f64 {
    let raw = prf(context, token_id, key) as f64;
    ((raw + 0.5) / 18_446_744_073_709_551_616.0).clamp(1e-15, 1.0 - 1e-15)
}

fn unique_ngrams(token_ids: &[i64], window: usize) -> Vec<(&[i64], i64)> {
    let mut seen: HashSet<&[i64]> = HashSet::new();
    let mut pairs = Vec::new();
    if window == 0 {
        return pairs;
    }
    for end in window - 1..token_ids.len() {
        let ngram = &token_ids[end + 1 - window..=end];
        if seen.insert(ngram) {
            pairs.push((&ngram[..window - 1], ngram[window - 1]));
        }
    }
    pairs
}

fn exp_race_score(
    scheme: WatermarkScheme,
    exp_sum: f64,
    scored: usize,
    threshold_z: f64,
) -> TextWatermarkScore {
    if scored == 0 {
        return make_score(scheme, 0.0, 0, threshold_z);
    }
    make_score(
        scheme,
        (exp_sum - scored as f64) / (scored as f64).sqrt(),
        scored,
        threshold_z,
    )
}

/// CPython 3.12 `sum()` over floats (Neumaier compensation).
fn neumaier_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let (mut total, mut compensation) = (0.0f64, 0.0f64);
    for value in values {
        let next = total + value;
        compensation += if total.abs() >= value.abs() {
            (total - next) + value
        } else {
            (value - next) + total
        };
        total = next;
    }
    if compensation != 0.0 && compensation.is_finite() {
        total + compensation
    } else {
        total
    }
}

// Wichura AS241 coefficients, verbatim from CPython's `statistics._normal_dist_inv_cdf`.
#[allow(
    clippy::excessive_precision,
    clippy::inconsistent_digit_grouping,
    clippy::unreadable_literal,
    clippy::large_digit_groups
)]
const CENTRAL_NUM: [f64; 8] = [
    2.50908_09287_30122_6727e+3,
    3.34305_75583_58812_8105e+4,
    6.72657_70927_00870_0853e+4,
    4.59219_53931_54987_1457e+4,
    1.37316_93765_50946_1125e+4,
    1.97159_09503_06551_4427e+3,
    1.33141_66789_17843_7745e+2,
    3.38713_28727_96366_6080e+0,
];
#[allow(
    clippy::excessive_precision,
    clippy::inconsistent_digit_grouping,
    clippy::unreadable_literal,
    clippy::large_digit_groups
)]
const CENTRAL_DEN: [f64; 8] = [
    5.22649_52788_52854_5610e+3,
    2.87290_85735_72194_2674e+4,
    3.93078_95800_09271_0610e+4,
    2.12137_94301_58659_5867e+4,
    5.39419_60214_24751_1077e+3,
    6.87187_00749_20579_0830e+2,
    4.23133_30701_60091_1252e+1,
    1.0,
];
#[allow(
    clippy::excessive_precision,
    clippy::inconsistent_digit_grouping,
    clippy::unreadable_literal,
    clippy::large_digit_groups
)]
const NEAR_NUM: [f64; 8] = [
    7.74545_01427_83414_07640e-4,
    2.27238_44989_26918_45833e-2,
    2.41780_72517_74506_11770e-1,
    1.27045_82524_52368_38258e+0,
    3.64784_83247_63204_60504e+0,
    5.76949_72214_60691_40550e+0,
    4.63033_78461_56545_29590e+0,
    1.42343_71107_49683_57734e+0,
];
#[allow(
    clippy::excessive_precision,
    clippy::inconsistent_digit_grouping,
    clippy::unreadable_literal,
    clippy::large_digit_groups
)]
const NEAR_DEN: [f64; 8] = [
    1.05075_00716_44416_84324e-9,
    5.47593_80849_95344_94600e-4,
    1.51986_66563_61645_71966e-2,
    1.48103_97642_74800_74590e-1,
    6.89767_33498_51000_04550e-1,
    1.67638_48301_83803_84940e+0,
    2.05319_16266_37758_82187e+0,
    1.0,
];
#[allow(
    clippy::excessive_precision,
    clippy::inconsistent_digit_grouping,
    clippy::unreadable_literal,
    clippy::large_digit_groups
)]
const TAIL_NUM: [f64; 8] = [
    2.01033_43992_92288_13265e-7,
    2.71155_55687_43487_57815e-5,
    1.24266_09473_88078_43860e-3,
    2.65321_89526_57612_30930e-2,
    2.96560_57182_85048_91230e-1,
    1.78482_65399_17291_33580e+0,
    5.46378_49111_64114_36990e+0,
    6.65790_46435_01103_77720e+0,
];
#[allow(
    clippy::excessive_precision,
    clippy::inconsistent_digit_grouping,
    clippy::unreadable_literal,
    clippy::large_digit_groups
)]
const TAIL_DEN: [f64; 8] = [
    2.04426_31033_89939_78564e-15,
    1.42151_17583_16445_88870e-7,
    1.84631_83175_10054_68180e-5,
    7.86869_13114_56132_59100e-4,
    1.48753_61290_85061_48525e-2,
    1.36929_88092_27358_05310e-1,
    5.99832_20655_58879_37690e-1,
    1.0,
];

fn horner(coefficients: &[f64; 8], r: f64) -> f64 {
    coefficients[1..]
        .iter()
        .fold(coefficients[0], |acc, &coefficient| acc * r + coefficient)
}

/// Wichura AS241, as `statistics.NormalDist().inv_cdf`.
fn normal_inv_cdf(p: f64) -> f64 {
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180_625 - q * q;
        return horner(&CENTRAL_NUM, r) * q / horner(&CENTRAL_DEN, r);
    }
    let r = if q <= 0.0 { p } else { 1.0 - p };
    let r = (-r.ln()).sqrt();
    let x = if r <= 5.0 {
        let r = r - 1.6;
        horner(&NEAR_NUM, r) / horner(&NEAR_DEN, r)
    } else {
        let r = r - 5.0;
        horner(&TAIL_NUM, r) / horner(&TAIL_DEN, r)
    };
    if q < 0.0 { -x } else { x }
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + libm::erf(x / std::f64::consts::SQRT_2))
}

fn require_context(context_len: usize) -> Result<(), TextError> {
    if context_len < 1 {
        return Err(TextError::invalid("context_len must be at least 1"));
    }
    Ok(())
}

/// Custom BLAKE2 Gumbel research simulation.
pub fn score_gumbel_text(
    token_ids: &[i64],
    secret_key: i64,
    context_len: usize,
    threshold_z: f64,
) -> Result<TextWatermarkScore, TextError> {
    require_context(context_len)?;
    let pairs = unique_ngrams(token_ids, context_len + 1);
    let exp_sum = neumaier_sum(
        pairs
            .iter()
            .map(|(context, token)| -(1.0 - prf_uniform(context, *token, secret_key)).ln()),
    );
    Ok(exp_race_score(
        WatermarkScheme::Gumbel,
        exp_sum,
        pairs.len(),
        threshold_z,
    ))
}

/// BLAKE2 block-column research simulation (not OpenAI textGrain).
pub fn score_textgrain_text(
    token_ids: &[i64],
    secret_key: i64,
    context_len: usize,
    num_blocks: usize,
    num_columns: usize,
    threshold_z: f64,
) -> Result<TextWatermarkScore, TextError> {
    require_context(context_len)?;
    if num_blocks < 2 || num_columns < 2 {
        return Err(TextError::invalid(
            "num_blocks and num_columns must be at least 2",
        ));
    }
    let pairs = unique_ngrams(token_ids, context_len + 1);
    let mut exp_sum = 0.0;
    for (context, token) in &pairs {
        let u_part = prf_uniform(context, *token, secret_key + TEXTGRAIN_SUB_PART);
        let block = ((u_part * num_blocks as f64) as usize).min(num_blocks - 1);
        let column = ((prf_uniform(context, 0, secret_key + TEXTGRAIN_SUB_COL) * num_columns as f64)
            as usize)
            .min(num_columns - 1);
        let cell = (block * num_columns + column) as i64;
        exp_sum += -(1.0 - prf_uniform(context, cell, secret_key + TEXTGRAIN_SUB_COST)).ln();
    }
    Ok(exp_race_score(
        WatermarkScheme::Textgrain,
        exp_sum,
        pairs.len(),
        threshold_z,
    ))
}

/// BLAKE2 two-key copula research simulation (not Meta's TextSeal2 detector).
pub fn score_textseal2_text(
    token_ids: &[i64],
    key_a: i64,
    key_b: Option<i64>,
    context_len: usize,
    threshold_z: f64,
) -> Result<TextWatermarkScore, TextError> {
    require_context(context_len)?;
    let pairs = unique_ngrams(token_ids, context_len + 1);
    let key_b = key_b.unwrap_or(key_a + TEXTSEAL_DUAL_KEY_OFFSET);
    let inv_sqrt2 = 1.0 / 2f64.sqrt();
    let mut exp_sum = 0.0;
    for (context, token) in &pairs {
        let u_a = prf_uniform(context, *token, key_a);
        let u_b = prf_uniform(context, *token, key_b);
        let g_star = (normal_inv_cdf(u_a) + normal_inv_cdf(u_b)) * inv_sqrt2;
        let phi_neg = normal_cdf(-g_star).clamp(1e-15, 1.0 - 1e-15);
        exp_sum += -phi_neg.ln();
    }
    Ok(exp_race_score(
        WatermarkScheme::Textseal2,
        exp_sum,
        pairs.len(),
        threshold_z,
    ))
}

fn score_greenlist(
    scheme: WatermarkScheme,
    token_ids: &[i64],
    secret_key: i64,
    context_width: usize,
    greenlist_ratio: f64,
    threshold_z: f64,
) -> Result<TextWatermarkScore, TextError> {
    if !(greenlist_ratio > 0.0 && greenlist_ratio < 1.0) {
        return Err(TextError::invalid("greenlist_ratio must be in (0, 1)"));
    }
    if token_ids.len() <= context_width {
        return Ok(make_score(scheme, 0.0, 0, threshold_z));
    }
    let cutoff = (greenlist_ratio * 18_446_744_073_709_551_616.0) as u128;
    let mut green = 0usize;
    let mut scored = 0usize;
    for index in context_width..token_ids.len() {
        if u128::from(prf(
            &token_ids[index - context_width..index],
            token_ids[index],
            secret_key,
        )) < cutoff
        {
            green += 1;
        }
        scored += 1;
    }
    let z_score = (green as f64 - greenlist_ratio * scored as f64)
        / (scored as f64 * greenlist_ratio * (1.0 - greenlist_ratio)).sqrt();
    Ok(make_score(scheme, z_score, scored, threshold_z))
}

/// BLAKE2 green-list simulation (not the HF/MarkLLM KGW detector).
pub fn score_kgw_text(
    token_ids: &[i64],
    secret_key: i64,
    context_width: usize,
    greenlist_ratio: f64,
    threshold_z: f64,
) -> Result<TextWatermarkScore, TextError> {
    if context_width < 1 {
        return Err(TextError::invalid("context_width must be at least 1"));
    }
    score_greenlist(
        WatermarkScheme::Kgw,
        token_ids,
        secret_key,
        context_width,
        greenlist_ratio,
        threshold_z,
    )
}

/// BLAKE2 context-free simulation (not the reference Unigram detector).
pub fn score_unigram_text(
    token_ids: &[i64],
    secret_key: i64,
    greenlist_ratio: f64,
    threshold_z: f64,
) -> Result<TextWatermarkScore, TextError> {
    score_greenlist(
        WatermarkScheme::Unigram,
        token_ids,
        secret_key,
        0,
        greenlist_ratio,
        threshold_z,
    )
}

/// Settings for [`score_synthid_threefry_text`]; defaults follow the reference configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct ThreefryOptions {
    pub ngram_len: usize,
    pub depth: usize,
    pub context_history_size: usize,
    pub eos_token_ids: Vec<i64>,
    pub alpha: f64,
}

impl Default for ThreefryOptions {
    fn default() -> Self {
        Self {
            ngram_len: 5,
            depth: 30,
            context_history_size: 1024,
            eos_token_ids: Vec::new(),
            alpha: 0.01,
        }
    }
}

/// Score the open single-key Threefry SynthID reference with a binomial tail.
///
/// Tokens from the first EOS onward are excluded; repeated contexts are masked within the history.
pub fn score_synthid_threefry_text(
    token_ids: &[i64],
    key: u64,
    options: &ThreefryOptions,
) -> Result<SynthIDThreefryScore, TextError> {
    if key >= 1 << 63 {
        return Err(TextError::invalid("key must be an integer in [0, 2**63)"));
    }
    if options.ngram_len < 2 || options.depth < 1 || options.context_history_size < 1 {
        return Err(TextError::invalid(
            "ngram_len must be at least 2, and depth and context history must be positive",
        ));
    }
    if !(options.alpha > 0.0 && options.alpha < 1.0) {
        return Err(TextError::invalid("alpha must be between 0 and 1, exclusive"));
    }
    let mut continuation = Vec::with_capacity(token_ids.len());
    for &token in token_ids {
        if options.eos_token_ids.contains(&token) {
            break;
        }
        let token = u32::try_from(token)
            .map_err(|_| TextError::invalid("token IDs must fit an unsigned 32-bit word"))?;
        continuation.push(token);
    }
    let (ones, positions) = threefry::count_g_values(
        &continuation,
        key,
        options.ngram_len,
        options.depth,
        options.context_history_size,
    );
    let trials = positions * options.depth as u64;
    let p_value = threefry::binomial_half_survival(ones, trials);
    Ok(SynthIDThreefryScore {
        p_value,
        detected: p_value <= options.alpha,
        num_tokens: positions,
        num_ones: ones,
        num_trials: trials,
        alpha: options.alpha,
    })
}
