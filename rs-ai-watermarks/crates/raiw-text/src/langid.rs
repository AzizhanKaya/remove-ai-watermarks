//! py3langid 0.4.0 naive-Bayes language identification with `norm_probs=True`.
//!
//! The model is the exact `data/model.npz.xz` from the published wheel (pinned by SHA-256):
//! an explicit path (`RAIW_LANGID_MODEL`), the model cache, or a one-time PyPI download.

use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use half::f16;

use crate::error::TextError;
use crate::guard::LanguageClassifier;
use crate::unicode;

/// SHA-256 of `py3langid/data/model.npz.xz` in py3langid 0.4.0.
pub const MODEL_SHA256: &str = "f4f4a2c3465ca1f081541037f9cac23021d68151c2f709c55ae5eedfca522963";
const WHEEL_INDEX: &str = "https://pypi.org/pypi/py3langid/0.4.0/json";
const WHEEL_NAME: &str = "py3langid-0.4.0-py3-none-any.whl";
const WHEEL_MEMBER: &str = "py3langid/data/model.npz.xz";

fn model_error(message: impl std::fmt::Display) -> TextError {
    TextError::runtime(format!(
        "The local language identifier model could not be loaded: {message}"
    ))
}

/// One `.npy` array: dtype descriptor, shape and raw little-endian data.
struct NpyArray {
    descr: String,
    shape: Vec<usize>,
    data: Vec<u8>,
}

fn parse_npy(bytes: Vec<u8>) -> Result<NpyArray, TextError> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(model_error("bad npy magic"));
    }
    let (header_len, offset) = if bytes[6] == 1 {
        (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10)
    } else {
        let field = bytes
            .get(8..12)
            .ok_or_else(|| model_error("truncated npy header"))?;
        (
            u32::from_le_bytes([field[0], field[1], field[2], field[3]]) as usize,
            12,
        )
    };
    let header = std::str::from_utf8(
        bytes
            .get(offset..offset + header_len)
            .ok_or_else(|| model_error("truncated npy header"))?,
    )
    .map_err(model_error)?;
    let field = |name: &str| -> Result<&str, TextError> {
        let start = header
            .find(&format!("'{name}':"))
            .ok_or_else(|| model_error(format!("npy header lacks {name}")))?;
        Ok(header[start + name.len() + 3..].trim_start())
    };
    let descr = field("descr")?
        .trim_start_matches('\'')
        .split('\'')
        .next()
        .unwrap_or("")
        .to_owned();
    if field("fortran_order")?.starts_with("True") {
        return Err(model_error("fortran-ordered arrays are unsupported"));
    }
    let shape_text = field("shape")?;
    let shape_text = shape_text.trim_start_matches('(').split(')').next().unwrap_or("");
    let shape = shape_text
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| item.parse::<usize>().map_err(model_error))
        .collect::<Result<Vec<_>, _>>()?;
    let mut data = bytes;
    data.drain(..offset + header_len);
    Ok(NpyArray { descr, shape, data })
}

impl NpyArray {
    fn len(&self) -> usize {
        self.shape.iter().product()
    }

    fn check(&self, descr: &str, width: usize) -> Result<(), TextError> {
        if self.descr != descr || self.data.len() < self.len() * width {
            return Err(model_error(format!(
                "unexpected dtype {} for {descr}",
                self.descr
            )));
        }
        Ok(())
    }

    fn u32s(&self) -> Result<Vec<u32>, TextError> {
        match self.descr.as_str() {
            "<u4" => {
                self.check("<u4", 4)?;
                Ok(self
                    .data
                    .chunks_exact(4)
                    .take(self.len())
                    .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect())
            }
            "<u2" => {
                self.check("<u2", 2)?;
                Ok(self
                    .data
                    .chunks_exact(2)
                    .take(self.len())
                    .map(|b| u32::from(u16::from_le_bytes([b[0], b[1]])))
                    .collect())
            }
            other => Err(model_error(format!("unexpected index dtype {other}"))),
        }
    }

    fn i32s(&self) -> Result<Vec<i32>, TextError> {
        self.check("<i4", 4)?;
        Ok(self
            .data
            .chunks_exact(4)
            .take(self.len())
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }

    fn f32s(&self) -> Result<Vec<f32>, TextError> {
        self.check("<f4", 4)?;
        Ok(self
            .data
            .chunks_exact(4)
            .take(self.len())
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }

    fn f16s(&self) -> Result<Vec<f16>, TextError> {
        self.check("<f2", 2)?;
        Ok(self
            .data
            .chunks_exact(2)
            .take(self.len())
            .map(|b| f16::from_le_bytes([b[0], b[1]]))
            .collect())
    }

    fn unicode_strings(&self) -> Result<Vec<String>, TextError> {
        let width: usize = self
            .descr
            .strip_prefix("<U")
            .and_then(|n| n.parse().ok())
            .ok_or_else(|| model_error("bad label dtype"))?;
        self.check(&self.descr.clone(), width * 4)?;
        Ok(self
            .data
            .chunks_exact(width * 4)
            .take(self.len())
            .map(|item| {
                item.chunks_exact(4)
                    .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .take_while(|&cp| cp != 0)
                    .filter_map(char::from_u32)
                    .collect()
            })
            .collect())
    }
}

/// The loaded identifier.
#[derive(Debug)]
pub struct LanguageIdentifier {
    ptc: Vec<f16>,
    pc: Vec<f32>,
    classes: Vec<String>,
    nextmove: Vec<u32>,
    rowbase: Vec<usize>,
    output: Vec<i32>,
    alias_pairs: Vec<(usize, usize)>,
}

impl LanguageIdentifier {
    /// Parse `model.npz.xz` bytes.
    pub fn from_xz_bytes(compressed: &[u8]) -> Result<Self, TextError> {
        let mut npz = Vec::new();
        lzma_rust2::XzReader::new(Cursor::new(compressed), true)
            .read_to_end(&mut npz)
            .map_err(model_error)?;
        let mut archive = zip::ZipArchive::new(Cursor::new(npz)).map_err(model_error)?;
        let mut arrays = HashMap::new();
        for name in ["ptc", "pc", "classes", "nextmove", "nextmove_row", "out_feat"] {
            let mut entry = archive.by_name(&format!("{name}.npy")).map_err(model_error)?;
            let mut bytes = Vec::with_capacity(entry.size() as usize);
            entry.read_to_end(&mut bytes).map_err(model_error)?;
            arrays.insert(name, parse_npy(bytes)?);
        }
        let array = |name: &str| {
            arrays
                .get(name)
                .ok_or_else(|| model_error(format!("missing {name}")))
        };
        let ptc_array = array("ptc")?;
        let classes = array("classes")?.unicode_strings()?;
        if ptc_array.shape.get(1) != Some(&classes.len()) {
            return Err(model_error("class count mismatch"));
        }
        let rows = array("nextmove_row")?.u32s()?;
        let output = array("out_feat")?.i32s()?;
        if rows.len() != output.len() {
            return Err(model_error("one output slot per DFA state"));
        }
        let mut first: HashMap<&str, usize> = HashMap::new();
        let mut alias_pairs = Vec::new();
        for (index, class) in classes.iter().enumerate() {
            match first.get(class.as_str()) {
                Some(&original) => alias_pairs.push((original, index)),
                None => {
                    first.insert(class, index);
                }
            }
        }
        Ok(Self {
            ptc: ptc_array.f16s()?,
            pc: array("pc")?.f32s()?,
            nextmove: array("nextmove")?.u32s()?,
            rowbase: rows.iter().map(|&row| (row as usize) << 8).collect(),
            output,
            alias_pairs,
            classes,
        })
    }

    pub fn from_path(path: &Path) -> Result<Self, TextError> {
        let bytes = std::fs::read(path).map_err(|err| TextError::io(path, err))?;
        Self::from_xz_bytes(&bytes)
    }

    /// Per-class probabilities for `text` (`_decide` with `norm_probs=True`).
    pub fn scores(&self, text: &str) -> Vec<f32> {
        let text = if unicode::str_is_upper(text) {
            unicode::lower(text)
        } else {
            text.to_owned()
        };
        let bytes = unicode::nfc(&text).into_bytes();
        let classes = self.classes.len();
        let mut counts: Vec<(usize, u32)> = Vec::new();
        let mut index_of: HashMap<usize, usize> = HashMap::new();
        let mut state = 0usize;
        for &byte in &bytes {
            state = self
                .nextmove
                .get(self.rowbase.get(state).copied().unwrap_or(0) + usize::from(byte))
                .copied()
                .unwrap_or(0) as usize;
            if let Some(&feature) = self.output.get(state)
                && feature >= 0
            {
                let feature = feature as usize;
                if let Some(&slot) = index_of.get(&feature) {
                    counts[slot].1 += 1;
                } else {
                    index_of.insert(feature, counts.len());
                    counts.push((feature, 1));
                }
            }
        }
        let mut scores = vec![0f32; classes];
        if !counts.is_empty() {
            for (feature, count) in &counts {
                let weight = (*count as f32).ln_1p();
                let row = &self.ptc[feature * classes..(feature + 1) * classes];
                for (score, value) in scores.iter_mut().zip(row) {
                    *score += weight * value.to_f32();
                }
            }
            for (score, prior) in scores.iter_mut().zip(&self.pc) {
                *score += prior;
            }
        }
        // T = sqrt(bytes) keeps the softmax calibrated across lengths.
        let temperature = (1.0 / (bytes.len().max(1) as f64).sqrt()) as f32;
        let mut max = f32::NEG_INFINITY;
        for score in &mut scores {
            *score *= temperature;
            max = max.max(*score);
        }
        for score in &mut scores {
            *score = (*score - max).exp();
        }
        let total = pairwise_sum(&scores);
        for score in &mut scores {
            *score /= total;
        }
        for &(first, duplicate) in &self.alias_pairs {
            scores[first] += scores[duplicate];
            scores[duplicate] = 0.0;
        }
        scores
    }

    /// `(language, confidence)` for `text`.
    pub fn classify_text(&self, text: &str) -> (String, f64) {
        let scores = self.scores(text);
        let (index, score) =
            scores
                .iter()
                .enumerate()
                .fold((0, f32::NEG_INFINITY), |best, (index, &score)| {
                    if score > best.1 { (index, score) } else { best }
                });
        (
            self.classes.get(index).cloned().unwrap_or_default(),
            f64::from(score),
        )
    }
}

/// NumPy's float32 pairwise summation (`add.reduce` on a contiguous array).
fn pairwise_sum(values: &[f32]) -> f32 {
    let n = values.len();
    if n < 8 {
        return values.iter().fold(0.0, |acc, &value| acc + value);
    }
    if n <= 128 {
        let mut lanes = [0f32; 8];
        lanes.copy_from_slice(&values[..8]);
        let blocked = n - n % 8;
        for chunk in values[8..blocked].chunks_exact(8) {
            for (lane, &value) in lanes.iter_mut().zip(chunk) {
                *lane += value;
            }
        }
        let mut sum =
            ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        for &value in &values[blocked..] {
            sum += value;
        }
        return sum;
    }
    let half = n / 2;
    let half = half - half % 8;
    pairwise_sum(&values[..half]) + pairwise_sum(&values[half..])
}

impl LanguageClassifier for LanguageIdentifier {
    fn classify(&self, text: &str) -> Result<(String, f64), TextError> {
        Ok(self.classify_text(text))
    }
}

fn verify_model(bytes: &[u8]) -> Result<(), TextError> {
    let digest = raiw_core::sha256_bytes(bytes);
    if digest != MODEL_SHA256 {
        return Err(model_error(format!("unexpected model SHA-256 {digest}")));
    }
    Ok(())
}

fn cached_model_path() -> PathBuf {
    raiw_core::model_cache_dir()
        .join("py3langid-0.4.0")
        .join("model.npz.xz")
}

fn http_get(url: &str) -> Result<Vec<u8>, TextError> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_mins(5)))
        .build()
        .into();
    let mut response = agent.get(url).call().map_err(model_error)?;
    response
        .body_mut()
        .with_config()
        .limit(64 * 1024 * 1024)
        .read_to_vec()
        .map_err(model_error)
}

fn download_model() -> Result<Vec<u8>, TextError> {
    let index: serde_json::Value = serde_json::from_slice(&http_get(WHEEL_INDEX)?).map_err(model_error)?;
    let url = index["urls"]
        .as_array()
        .and_then(|files| files.iter().find(|file| file["filename"] == WHEEL_NAME))
        .and_then(|file| file["url"].as_str())
        .ok_or_else(|| model_error("py3langid 0.4.0 wheel not listed on PyPI"))?
        .to_owned();
    let wheel = http_get(&url)?;
    let mut archive = zip::ZipArchive::new(Cursor::new(wheel)).map_err(model_error)?;
    let mut entry = archive.by_name(WHEEL_MEMBER).map_err(model_error)?;
    let mut model = Vec::new();
    entry.read_to_end(&mut model).map_err(model_error)?;
    verify_model(&model)?;
    Ok(model)
}

/// Locate the pinned model bytes: `RAIW_LANGID_MODEL`, the cache, then PyPI.
pub fn load_model_bytes() -> Result<Vec<u8>, TextError> {
    if let Some(path) = std::env::var_os("RAIW_LANGID_MODEL").filter(|value| !value.is_empty()) {
        let bytes = std::fs::read(&path).map_err(|err| TextError::io(PathBuf::from(&path), err))?;
        verify_model(&bytes)?;
        return Ok(bytes);
    }
    let cached = cached_model_path();
    if let Ok(bytes) = std::fs::read(&cached)
        && verify_model(&bytes).is_ok()
    {
        return Ok(bytes);
    }
    let bytes = download_model()?;
    if let Some(parent) = cached.parent()
        && std::fs::create_dir_all(parent).is_ok()
        && let Ok(stage) = raiw_core::StagedSibling::new(&cached)
        && std::fs::write(stage.path(), &bytes).is_ok()
    {
        let _ = stage.commit();
    }
    Ok(bytes)
}

static IDENTIFIER: OnceLock<Result<LanguageIdentifier, String>> = OnceLock::new();

/// The process-wide identifier, loaded once; user text is never cached.
pub fn language_identifier() -> Result<&'static LanguageIdentifier, TextError> {
    IDENTIFIER
        .get_or_init(|| {
            tracing::info!("Loading the local text language identifier");
            load_model_bytes()
                .and_then(|bytes| LanguageIdentifier::from_xz_bytes(&bytes))
                .map_err(|err| err.to_string())
        })
        .as_ref()
        .map_err(|message| TextError::runtime(message.clone()))
}

/// The default classifier backed by [`language_identifier`].
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultLanguageClassifier;

impl LanguageClassifier for DefaultLanguageClassifier {
    fn classify(&self, text: &str) -> Result<(String, f64), TextError> {
        language_identifier().map(|identifier| identifier.classify_text(text))
    }
}
