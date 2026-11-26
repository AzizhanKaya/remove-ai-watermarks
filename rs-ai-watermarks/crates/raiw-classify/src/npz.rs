//! Pickle-free `.npz` reading: a zip of `.npy` members (stored or deflated), as `np.load(allow_pickle=False)`.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use half::f16;
use zip::ZipArchive;

use crate::error::{ClassifyError, Result};

/// Typed payload of one `.npy` array, in C order.
#[derive(Debug, Clone, PartialEq)]
pub enum NpyData {
    F16(Vec<f16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    Bool(Vec<bool>),
    Str(Vec<String>),
}

/// One decoded `.npy` array.
#[derive(Debug, Clone, PartialEq)]
pub struct NpyArray {
    pub shape: Vec<usize>,
    pub data: NpyData,
}

/// The `.npy` header fields (`descr`, `fortran_order`, `shape`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NpyHeader {
    pub descr: String,
    pub fortran_order: bool,
    pub shape: Vec<usize>,
}

impl NpyHeader {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

macro_rules! numeric_cast {
    ($self:ident, $t:ty) => {
        match &$self.data {
            NpyData::F16(v) => v.iter().map(|x| x.to_f64() as $t).collect(),
            NpyData::F32(v) => v.iter().map(|&x| x as $t).collect(),
            NpyData::F64(v) => v.iter().map(|&x| x as $t).collect(),
            NpyData::I8(v) => v.iter().map(|&x| x as $t).collect(),
            NpyData::I16(v) => v.iter().map(|&x| x as $t).collect(),
            NpyData::I32(v) => v.iter().map(|&x| x as $t).collect(),
            NpyData::I64(v) => v.iter().map(|&x| x as $t).collect(),
            NpyData::U8(v) => v.iter().map(|&x| x as $t).collect(),
            NpyData::U16(v) => v.iter().map(|&x| x as $t).collect(),
            NpyData::U32(v) => v.iter().map(|&x| x as $t).collect(),
            NpyData::U64(v) => v.iter().map(|&x| x as $t).collect(),
            NpyData::Bool(v) => v.iter().map(|&x| u8::from(x) as $t).collect(),
            NpyData::Str(_) => return Err(ClassifyError::bundle("expected a numeric array, found strings")),
        }
    };
}

impl NpyArray {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// `np.asarray(a, dtype=np.float64)`.
    pub fn to_f64(&self) -> Result<Vec<f64>> {
        Ok(numeric_cast!(self, f64))
    }

    /// `np.asarray(a, dtype=np.float32)` (float64 rounds to nearest).
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        Ok(match &self.data {
            NpyData::F16(v) => v.iter().map(|x| x.to_f32()).collect(),
            _ => numeric_cast!(self, f32),
        })
    }

    /// `np.asarray(a, dtype=np.int64)` (floats truncate toward zero).
    pub fn to_i64(&self) -> Result<Vec<i64>> {
        Ok(numeric_cast!(self, i64))
    }

    pub fn is_float(&self) -> bool {
        matches!(self.data, NpyData::F16(_) | NpyData::F32(_) | NpyData::F64(_))
    }

    /// `[str(x) for x in np.asarray(a).tolist()]` for unicode arrays.
    pub fn strings(&self) -> Result<Vec<String>> {
        match &self.data {
            NpyData::Str(v) => Ok(v.clone()),
            _ => Err(ClassifyError::bundle("expected a unicode string array")),
        }
    }

    /// Raw bytes of a `uint8` array.
    pub fn bytes(&self) -> Option<&[u8]> {
        match &self.data {
            NpyData::U8(v) => Some(v),
            _ => None,
        }
    }
}

/// An open `.npz` archive whose members decode on demand.
#[derive(Debug)]
pub struct Npz {
    path: PathBuf,
    archive: ZipArchive<BufReader<File>>,
    keys: BTreeSet<String>,
}

impl Npz {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|err| ClassifyError::io(path, err))?;
        let archive = ZipArchive::new(BufReader::new(file)).map_err(|err| {
            ClassifyError::bundle(format!("{} is not a valid .npz archive: {err}", path.display()))
        })?;
        let keys = archive
            .file_names()
            .filter_map(|name| name.ok()?.strip_suffix(".npy").map(str::to_owned))
            .collect();
        Ok(Self {
            path: path.to_path_buf(),
            archive,
            keys,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// `key in data`.
    pub fn contains(&self, key: &str) -> bool {
        self.keys.contains(key)
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.keys.iter().map(String::as_str)
    }

    /// `data[key]`; a missing key is a `KeyError` in Python, reported here as an invalid bundle.
    pub fn get(&mut self, key: &str) -> Result<NpyArray> {
        let mut member = self.member(key)?;
        let header = read_header(&mut member, key)?;
        let mut body = Vec::new();
        member
            .read_to_end(&mut body)
            .map_err(|err| ClassifyError::bundle(format!("cannot read {key}.npy: {err}")))?;
        decode_body(&header, &body, key)
    }

    /// Only the header of `key` (Python's `_inspect_npz_1d_uint8_length`).
    pub fn header(&mut self, key: &str) -> Result<NpyHeader> {
        let mut member = self.member(key)?;
        read_header(&mut member, key)
    }

    fn member(&mut self, key: &str) -> Result<zip::read::ZipFile<'_, BufReader<File>>> {
        if !self.keys.contains(key) {
            return Err(ClassifyError::bundle(format!(
                "'{key} is not a file in the archive'"
            )));
        }
        self.archive
            .by_name(&format!("{key}.npy"))
            .map_err(|err| ClassifyError::bundle(format!("cannot open {key}.npy: {err}")))
    }
}

fn read_header(reader: &mut impl Read, key: &str) -> Result<NpyHeader> {
    let bad = |what: &str| ClassifyError::bundle(format!("invalid .npy header for {key}: {what}"));
    let mut magic = [0u8; 8];
    reader
        .read_exact(&mut magic)
        .map_err(|_| bad("truncated magic"))?;
    if &magic[..6] != b"\x93NUMPY" {
        return Err(bad("bad magic"));
    }
    let header_len = match magic[6] {
        1 => {
            let mut b = [0u8; 2];
            reader.read_exact(&mut b).map_err(|_| bad("truncated length"))?;
            usize::from(u16::from_le_bytes(b))
        }
        2 | 3 => {
            let mut b = [0u8; 4];
            reader.read_exact(&mut b).map_err(|_| bad("truncated length"))?;
            u32::from_le_bytes(b) as usize
        }
        _ => return Err(bad("unsupported version")),
    };
    let mut text = vec![0u8; header_len];
    reader
        .read_exact(&mut text)
        .map_err(|_| bad("truncated dictionary"))?;
    let text = String::from_utf8_lossy(&text);
    parse_header_dict(&text).ok_or_else(|| bad(&text))
}

fn parse_header_dict(text: &str) -> Option<NpyHeader> {
    let descr = {
        let at = text.find("'descr'")?;
        let rest = &text[at + 7..];
        let open = rest.find('\'')?;
        let rest = &rest[open + 1..];
        let close = rest.find('\'')?;
        rest[..close].to_owned()
    };
    let fortran_order = {
        let at = text.find("'fortran_order'")?;
        text[at + 15..].trim_start_matches([':', ' ']).starts_with("True")
    };
    let shape = {
        let at = text.find("'shape'")?;
        let rest = &text[at + 7..];
        let open = rest.find('(')?;
        let close = rest.find(')')?;
        rest[open + 1..close]
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.trim_end_matches('L').parse::<usize>().ok())
            .collect::<Option<Vec<_>>>()?
    };
    Some(NpyHeader {
        descr,
        fortran_order,
        shape,
    })
}

fn decode_body(header: &NpyHeader, body: &[u8], key: &str) -> Result<NpyArray> {
    let n = header.len();
    let descr = header.descr.as_str();
    let (order, kind) = descr.split_at(1);
    if order == ">" && !matches!(kind, "u1" | "i1" | "b1") {
        return Err(ClassifyError::bundle(format!(
            "big-endian array {key} is not supported"
        )));
    }
    if order == "|" && kind == "O" || kind.starts_with('O') {
        return Err(ClassifyError::bundle(format!(
            "Object arrays cannot be loaded when allow_pickle=False ({key})"
        )));
    }
    let need = |width: usize| -> Result<&[u8]> {
        body.get(..n * width)
            .ok_or_else(|| ClassifyError::bundle(format!("truncated array body for {key}")))
    };
    macro_rules! le {
        ($t:ty, $w:expr) => {
            need($w)?
                .chunks_exact($w)
                .map(|c| <$t>::from_le_bytes(c.try_into().unwrap_or([0; $w])))
                .collect()
        };
    }
    let data = match kind {
        "f2" => NpyData::F16(
            need(2)?
                .chunks_exact(2)
                .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
        ),
        "f4" => NpyData::F32(le!(f32, 4)),
        "f8" => NpyData::F64(le!(f64, 8)),
        "i1" => NpyData::I8(need(1)?.iter().map(|&b| b as i8).collect()),
        "i2" => NpyData::I16(le!(i16, 2)),
        "i4" => NpyData::I32(le!(i32, 4)),
        "i8" => NpyData::I64(le!(i64, 8)),
        "u1" => NpyData::U8(need(1)?.to_vec()),
        "u2" => NpyData::U16(le!(u16, 2)),
        "u4" => NpyData::U32(le!(u32, 4)),
        "u8" => NpyData::U64(le!(u64, 8)),
        "b1" => NpyData::Bool(need(1)?.iter().map(|&b| b != 0).collect()),
        _ if kind.starts_with('U') => {
            let chars: usize = kind[1..]
                .parse()
                .map_err(|_| ClassifyError::bundle(format!("bad unicode dtype {descr} for {key}")))?;
            let raw = need(chars * 4)?;
            let strings = if chars == 0 {
                vec![String::new(); n]
            } else {
                raw.chunks_exact(chars * 4)
                    .map(|item| {
                        item.chunks_exact(4)
                            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .take_while(|&cp| cp != 0)
                            .filter_map(char::from_u32)
                            .collect::<String>()
                    })
                    .collect()
            };
            NpyData::Str(strings)
        }
        _ => {
            return Err(ClassifyError::bundle(format!(
                "unsupported dtype {descr} for {key}"
            )));
        }
    };
    let mut array = NpyArray {
        shape: header.shape.clone(),
        data,
    };
    if header.fortran_order && array.ndim() > 1 {
        array = to_c_order(array);
    }
    Ok(array)
}

fn to_c_order(array: NpyArray) -> NpyArray {
    let shape = array.shape.clone();
    let n: usize = shape.iter().product();
    let perm: Vec<usize> = (0..n)
        .map(|c_index| {
            let mut rem = c_index;
            let mut coords = vec![0usize; shape.len()];
            for (d, &len) in shape.iter().enumerate().rev() {
                coords[d] = rem % len;
                rem /= len;
            }
            let mut f_index = 0;
            let mut stride = 1;
            for (d, &len) in shape.iter().enumerate() {
                f_index += coords[d] * stride;
                stride *= len;
            }
            f_index
        })
        .collect();
    macro_rules! permute {
        ($v:expr, $variant:ident) => {
            NpyData::$variant(perm.iter().map(|&i| $v[i].clone()).collect())
        };
    }
    let data = match &array.data {
        NpyData::F16(v) => permute!(v, F16),
        NpyData::F32(v) => permute!(v, F32),
        NpyData::F64(v) => permute!(v, F64),
        NpyData::I8(v) => permute!(v, I8),
        NpyData::I16(v) => permute!(v, I16),
        NpyData::I32(v) => permute!(v, I32),
        NpyData::I64(v) => permute!(v, I64),
        NpyData::U8(v) => permute!(v, U8),
        NpyData::U16(v) => permute!(v, U16),
        NpyData::U32(v) => permute!(v, U32),
        NpyData::U64(v) => permute!(v, U64),
        NpyData::Bool(v) => permute!(v, Bool),
        NpyData::Str(v) => permute!(v, Str),
    };
    NpyArray { shape, data }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_dictionary_parses_numpy_layouts() {
        let h = parse_header_dict("{'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), }").unwrap();
        assert_eq!(
            h,
            NpyHeader {
                descr: "<f8".into(),
                fortran_order: false,
                shape: vec![3, 4]
            }
        );
        let s = parse_header_dict("{'descr': '<U25', 'fortran_order': False, 'shape': (), }").unwrap();
        assert!(s.shape.is_empty());
        assert_eq!(s.len(), 1);
        let one = parse_header_dict("{'descr': '|u1', 'fortran_order': True, 'shape': (7,), }").unwrap();
        assert_eq!(one.shape, vec![7]);
        assert!(one.fortran_order);
    }

    #[test]
    fn unicode_and_fortran_bodies_decode() {
        let h = NpyHeader {
            descr: "<U3".into(),
            fortran_order: false,
            shape: vec![2],
        };
        let mut body = Vec::new();
        for s in ["ab", "xyz"] {
            let mut cps: Vec<u32> = s.chars().map(u32::from).collect();
            cps.resize(3, 0);
            cps.iter()
                .for_each(|cp| body.extend_from_slice(&cp.to_le_bytes()));
        }
        assert_eq!(
            decode_body(&h, &body, "k").unwrap().strings().unwrap(),
            vec!["ab", "xyz"]
        );

        let h = NpyHeader {
            descr: "<i8".into(),
            fortran_order: true,
            shape: vec![2, 3],
        };
        let body: Vec<u8> = [1i64, 4, 2, 5, 3, 6]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(
            decode_body(&h, &body, "k").unwrap().to_i64().unwrap(),
            vec![1, 2, 3, 4, 5, 6]
        );
    }
}
