//! Tolerant reader for `torch.save` zip checkpoints whose pickles carry arbitrary objects.
//!
//! burn-store's reader rejects dictionaries keyed by non-strings, which training
//! frameworks (omegaconf configs, optimizer state) routinely embed next to the weights.
//! This VM keeps every unknown object opaque and only materializes tensors.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::error::{GpuError, Result};
use crate::weights::HostTensor;

#[derive(Debug, Clone)]
enum Value {
    None,
    Int(i64),
    Str(String),
    Tuple(Vec<Value>),
    List(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Global(String, String),
    Storage {
        key: String,
        dtype: StorageType,
    },
    Tensor(TensorRef),
    /// Floats, bools, bytes and objects the weights never depend on.
    Opaque,
    Mark,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StorageType {
    F32,
    F64,
    F16,
    Bf16,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl StorageType {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "FloatStorage" => Self::F32,
            "DoubleStorage" => Self::F64,
            "HalfStorage" => Self::F16,
            "BFloat16Storage" => Self::Bf16,
            "LongStorage" => Self::I64,
            "IntStorage" => Self::I32,
            "ShortStorage" => Self::I16,
            "CharStorage" => Self::I8,
            "ByteStorage" => Self::U8,
            "BoolStorage" => Self::Bool,
            _ => return None,
        })
    }

    fn size(self) -> usize {
        match self {
            Self::F64 | Self::I64 => 8,
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::Bf16 | Self::I16 => 2,
            Self::I8 | Self::U8 | Self::Bool => 1,
        }
    }

    fn decode(self, bytes: &[u8]) -> f32 {
        let take = |n: usize| -> [u8; 8] {
            let mut buf = [0u8; 8];
            buf[..n].copy_from_slice(&bytes[..n]);
            buf
        };
        match self {
            Self::F32 => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            Self::F64 => f64::from_le_bytes(take(8)) as f32,
            Self::F16 => half::f16::from_le_bytes([bytes[0], bytes[1]]).to_f32(),
            Self::Bf16 => half::bf16::from_le_bytes([bytes[0], bytes[1]]).to_f32(),
            Self::I64 => i64::from_le_bytes(take(8)) as f32,
            Self::I32 => i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f32,
            Self::I16 => f32::from(i16::from_le_bytes([bytes[0], bytes[1]])),
            Self::I8 => f32::from(bytes[0] as i8),
            Self::U8 | Self::Bool => f32::from(bytes[0]),
        }
    }
}

#[derive(Debug, Clone)]
struct TensorRef {
    key: String,
    dtype: StorageType,
    offset: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

/// Read every tensor under `top_level_key` (or the root) keyed by dotted path.
pub(crate) fn read_checkpoint(
    path: &Path,
    top_level_key: Option<&str>,
) -> Result<BTreeMap<String, HostTensor>> {
    let fail = |reason: String| GpuError::Weights {
        path: path.to_path_buf(),
        reason,
    };
    let file = File::open(path).map_err(|err| GpuError::io(path, err))?;
    let mut archive = zip::ZipArchive::new(BufReader::new(file)).map_err(|err| fail(err.to_string()))?;
    let names: Vec<String> = archive
        .file_names()
        .filter_map(std::result::Result::ok)
        .map(|name| name.into_owned())
        .collect();
    let pickle_name = names
        .iter()
        .find(|name| name.ends_with("data.pkl"))
        .ok_or_else(|| fail("no data.pkl".to_owned()))?;
    let prefix = pickle_name.trim_end_matches("data.pkl").to_owned();
    let pickle = read_entry(&mut archive, pickle_name).map_err(fail)?;
    let root = Machine::default().run(&pickle).map_err(fail)?;
    let root = match top_level_key {
        Some(key) => lookup(&root, key).ok_or_else(|| fail(format!("no top-level key {key:?}")))?,
        None => root,
    };
    let mut refs = Vec::new();
    collect_tensors(&root, String::new(), &mut refs);
    let mut storages: HashMap<String, Vec<u8>> = HashMap::new();
    let mut tensors = BTreeMap::new();
    for (name, tensor) in refs {
        if !storages.contains_key(&tensor.key) {
            let bytes = read_entry(&mut archive, &format!("{prefix}data/{}", tensor.key)).map_err(&fail)?;
            storages.insert(tensor.key.clone(), bytes);
        }
        let bytes = &storages[&tensor.key];
        let data =
            gather(bytes, &tensor).ok_or_else(|| fail(format!("{name}: tensor view exceeds its storage")))?;
        tensors.insert(name, HostTensor::new(tensor.shape.clone(), data)?);
    }
    Ok(tensors)
}

fn read_entry<R: Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
    name: &str,
) -> std::result::Result<Vec<u8>, String> {
    let mut entry = archive.by_name(name).map_err(|err| format!("{name}: {err}"))?;
    let mut bytes = Vec::new();
    entry
        .read_to_end(&mut bytes)
        .map_err(|err| format!("{name}: {err}"))?;
    Ok(bytes)
}

fn gather(bytes: &[u8], tensor: &TensorRef) -> Option<Vec<f32>> {
    let size = tensor.dtype.size();
    let count: usize = tensor.shape.iter().product();
    let mut out = Vec::with_capacity(count);
    let mut index = vec![0usize; tensor.shape.len()];
    for _ in 0..count {
        let element = tensor.offset
            + index
                .iter()
                .zip(&tensor.stride)
                .map(|(i, s)| i * s)
                .sum::<usize>();
        let start = element * size;
        out.push(tensor.dtype.decode(bytes.get(start..start + size)?));
        for dim in (0..index.len()).rev() {
            index[dim] += 1;
            if index[dim] < tensor.shape[dim] {
                break;
            }
            index[dim] = 0;
        }
    }
    Some(out)
}

fn lookup(value: &Value, key: &str) -> Option<Value> {
    match value {
        Value::Dict(items) => items
            .iter()
            .find(|(k, _)| matches!(k, Value::Str(s) if s == key))
            .map(|(_, v)| v.clone()),
        _ => None,
    }
}

fn collect_tensors(value: &Value, path: String, out: &mut Vec<(String, TensorRef)>) {
    let join = |segment: &str| {
        if path.is_empty() {
            segment.to_owned()
        } else {
            format!("{path}.{segment}")
        }
    };
    match value {
        Value::Tensor(tensor) => out.push((path, tensor.clone())),
        Value::Dict(items) => {
            for (key, item) in items {
                let segment = match key {
                    Value::Str(s) => s.clone(),
                    Value::Int(i) => i.to_string(),
                    _ => continue,
                };
                collect_tensors(item, join(&segment), out);
            }
        }
        Value::List(items) | Value::Tuple(items) => {
            for (i, item) in items.iter().enumerate() {
                collect_tensors(item, join(&i.to_string()), out);
            }
        }
        _ => {}
    }
}

#[derive(Default)]
struct Machine {
    stack: Vec<Value>,
    memo: HashMap<u32, Value>,
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> std::result::Result<&'a [u8], String> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= self.data.len())
            .ok_or("truncated pickle")?;
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> std::result::Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> std::result::Result<u16, String> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> std::result::Result<u32, String> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn line(&mut self) -> std::result::Result<String, String> {
        let rest = &self.data[self.pos..];
        let end = rest
            .iter()
            .position(|&b| b == b'\n')
            .ok_or("unterminated pickle line")?;
        self.pos += end + 1;
        Ok(String::from_utf8_lossy(&rest[..end]).into_owned())
    }

    fn utf8(&mut self, n: usize) -> std::result::Result<String, String> {
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
}

impl Machine {
    fn pop(&mut self) -> std::result::Result<Value, String> {
        self.stack
            .pop()
            .ok_or_else(|| "pickle stack underflow".to_owned())
    }

    fn pop_mark(&mut self) -> std::result::Result<Vec<Value>, String> {
        let mark = self
            .stack
            .iter()
            .rposition(|v| matches!(v, Value::Mark))
            .ok_or("pickle mark missing")?;
        let items = self.stack.split_off(mark + 1);
        self.stack.pop();
        Ok(items)
    }

    fn top(&mut self) -> std::result::Result<&mut Value, String> {
        self.stack
            .last_mut()
            .ok_or_else(|| "pickle stack underflow".to_owned())
    }

    fn run(mut self, data: &[u8]) -> std::result::Result<Value, String> {
        let mut cur = Cursor { data, pos: 0 };
        loop {
            let op = cur.u8()?;
            match op {
                0x80 => {
                    cur.u8()?;
                }
                0x95 => {
                    cur.take(8)?;
                }
                b'.' => return self.pop(),
                b'(' => self.stack.push(Value::Mark),
                b'}' => self.stack.push(Value::Dict(Vec::new())),
                b']' => self.stack.push(Value::List(Vec::new())),
                b')' => self.stack.push(Value::Tuple(Vec::new())),
                b'N' => self.stack.push(Value::None),
                0x88 | 0x89 => self.stack.push(Value::Opaque),
                b'K' => {
                    let v = cur.u8()?;
                    self.stack.push(Value::Int(i64::from(v)));
                }
                b'M' => {
                    let v = cur.u16()?;
                    self.stack.push(Value::Int(i64::from(v)));
                }
                b'J' => {
                    let v = cur.u32()? as i32;
                    self.stack.push(Value::Int(i64::from(v)));
                }
                0x8a => {
                    let n = usize::from(cur.u8()?);
                    let bytes = cur.take(n)?;
                    self.stack.push(long_value(bytes));
                }
                0x8b => {
                    let n = cur.u32()? as usize;
                    let bytes = cur.take(n)?;
                    self.stack.push(long_value(bytes));
                }
                b'I' => {
                    let line = cur.line()?;
                    let value = match line.as_str() {
                        "01" | "00" => Value::Opaque,
                        text => Value::Int(text.parse().map_err(|_| format!("bad INT {text:?}"))?),
                    };
                    self.stack.push(value);
                }
                b'L' => {
                    let line = cur.line()?;
                    let value = line
                        .trim_end_matches('L')
                        .parse()
                        .map_or(Value::Opaque, Value::Int);
                    self.stack.push(value);
                }
                b'G' => {
                    cur.take(8)?;
                    self.stack.push(Value::Opaque);
                }
                b'F' => {
                    cur.line()?;
                    self.stack.push(Value::Opaque);
                }
                b'X' => {
                    let n = cur.u32()? as usize;
                    let s = cur.utf8(n)?;
                    self.stack.push(Value::Str(s));
                }
                0x8c => {
                    let n = usize::from(cur.u8()?);
                    let s = cur.utf8(n)?;
                    self.stack.push(Value::Str(s));
                }
                0x8d => {
                    let b = cur.take(8)?;
                    let n = u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as usize;
                    let s = cur.utf8(n)?;
                    self.stack.push(Value::Str(s));
                }
                b'V' => {
                    let line = cur.line()?;
                    self.stack.push(Value::Str(line));
                }
                b'T' | b'B' => {
                    let n = cur.u32()? as usize;
                    cur.take(n)?;
                    self.stack.push(Value::Opaque);
                }
                b'U' | b'C' => {
                    let n = usize::from(cur.u8()?);
                    cur.take(n)?;
                    self.stack.push(Value::Opaque);
                }
                b'c' => {
                    let module = cur.line()?;
                    let name = cur.line()?;
                    self.stack.push(Value::Global(module, name));
                }
                0x93 => {
                    let name = self.pop()?;
                    let module = self.pop()?;
                    match (module, name) {
                        (Value::Str(module), Value::Str(name)) => {
                            self.stack.push(Value::Global(module, name))
                        }
                        _ => return Err("STACK_GLOBAL expects two strings".to_owned()),
                    }
                }
                b'q' => {
                    let idx = u32::from(cur.u8()?);
                    self.memoize(idx)?;
                }
                b'r' => {
                    let idx = cur.u32()?;
                    self.memoize(idx)?;
                }
                0x94 => {
                    let idx = self.memo.len() as u32;
                    self.memoize(idx)?;
                }
                b'h' => {
                    let idx = u32::from(cur.u8()?);
                    self.recall(idx)?;
                }
                b'j' => {
                    let idx = cur.u32()?;
                    self.recall(idx)?;
                }
                b't' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Value::Tuple(items));
                }
                0x85..=0x87 => {
                    let n = usize::from(op - 0x84);
                    if self.stack.len() < n {
                        return Err("pickle stack underflow".to_owned());
                    }
                    let items = self.stack.split_off(self.stack.len() - n);
                    self.stack.push(Value::Tuple(items));
                }
                b'l' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Value::List(items));
                }
                b'd' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Value::Dict(pairs(items)));
                }
                b'a' => {
                    let item = self.pop()?;
                    if let Value::List(list) = self.top()? {
                        list.push(item);
                    }
                }
                b'e' => {
                    let items = self.pop_mark()?;
                    if let Value::List(list) = self.top()? {
                        list.extend(items);
                    }
                }
                b's' => {
                    let value = self.pop()?;
                    let key = self.pop()?;
                    if let Value::Dict(dict) = self.top()? {
                        dict.push((key, value));
                    }
                }
                b'u' => {
                    let items = self.pop_mark()?;
                    if let Value::Dict(dict) = self.top()? {
                        dict.extend(pairs(items));
                    }
                }
                b'R' => {
                    let args = self.pop()?;
                    let callable = self.pop()?;
                    let value = reduce(callable, args)?;
                    self.stack.push(value);
                }
                0x81 => {
                    self.pop()?;
                    self.pop()?;
                    self.stack.push(Value::Opaque);
                }
                0x92 => {
                    self.pop()?;
                    self.pop()?;
                    self.pop()?;
                    self.stack.push(Value::Opaque);
                }
                b'b' => {
                    let _state = self.pop()?;
                }
                b'Q' => {
                    let pid = self.pop()?;
                    self.stack.push(persistent(pid)?);
                }
                b'0' => {
                    self.pop()?;
                }
                b'2' => {
                    let top = self.top()?.clone();
                    self.stack.push(top);
                }
                other => return Err(format!("unsupported pickle opcode 0x{other:02x}")),
            }
        }
    }

    fn memoize(&mut self, idx: u32) -> std::result::Result<(), String> {
        let top = self.top()?.clone();
        self.memo.insert(idx, top);
        Ok(())
    }

    fn recall(&mut self, idx: u32) -> std::result::Result<(), String> {
        let value = self
            .memo
            .get(&idx)
            .cloned()
            .ok_or_else(|| format!("pickle memo {idx} missing"))?;
        self.stack.push(value);
        Ok(())
    }
}

fn long_value(bytes: &[u8]) -> Value {
    if bytes.len() > 8 {
        return Value::Opaque;
    }
    let negative = bytes.last().is_some_and(|b| b & 0x80 != 0);
    let mut buf = if negative { [0xff; 8] } else { [0; 8] };
    buf[..bytes.len()].copy_from_slice(bytes);
    Value::Int(i64::from_le_bytes(buf))
}

fn pairs(items: Vec<Value>) -> Vec<(Value, Value)> {
    let mut iter = items.into_iter();
    let mut out = Vec::new();
    while let (Some(k), Some(v)) = (iter.next(), iter.next()) {
        out.push((k, v));
    }
    out
}

fn as_usize(value: &Value) -> std::result::Result<usize, String> {
    match value {
        Value::Int(i) => usize::try_from(*i).map_err(|_| format!("negative tensor geometry {i}")),
        _ => Err("tensor geometry must be integers".to_owned()),
    }
}

fn as_usizes(value: &Value) -> std::result::Result<Vec<usize>, String> {
    match value {
        Value::Tuple(items) | Value::List(items) => items.iter().map(as_usize).collect(),
        _ => Err("tensor size/stride must be a tuple".to_owned()),
    }
}

fn reduce(callable: Value, args: Value) -> std::result::Result<Value, String> {
    let Value::Global(module, name) = &callable else {
        return Ok(Value::Opaque);
    };
    let args = match args {
        Value::Tuple(items) => items,
        _ => Vec::new(),
    };
    match (module.as_str(), name.as_str()) {
        ("torch._utils", "_rebuild_tensor_v2" | "_rebuild_tensor") => {
            let [storage, offset, size, stride, ..] = args.as_slice() else {
                return Err("_rebuild_tensor_v2 needs storage, offset, size, stride".to_owned());
            };
            let Value::Storage { key, dtype } = storage else {
                return Err("_rebuild_tensor_v2 without a storage".to_owned());
            };
            Ok(Value::Tensor(TensorRef {
                key: key.clone(),
                dtype: *dtype,
                offset: as_usize(offset)?,
                shape: as_usizes(size)?,
                stride: as_usizes(stride)?,
            }))
        }
        ("torch._utils", "_rebuild_parameter" | "_rebuild_parameter_with_state") => {
            Ok(args.into_iter().next().unwrap_or(Value::None))
        }
        ("collections", "OrderedDict") | ("builtins" | "__builtin__", "dict") => Ok(Value::Dict(Vec::new())),
        _ => Ok(Value::Opaque),
    }
}

fn persistent(pid: Value) -> std::result::Result<Value, String> {
    let Value::Tuple(items) = pid else {
        return Err("persistent id must be a tuple".to_owned());
    };
    match items.as_slice() {
        [Value::Str(tag), Value::Global(_, storage), Value::Str(key), ..] if tag == "storage" => {
            let dtype =
                StorageType::from_name(storage).ok_or_else(|| format!("unsupported storage {storage}"))?;
            Ok(Value::Storage {
                key: key.clone(),
                dtype,
            })
        }
        _ => Err("unsupported persistent id".to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_decoding() {
        assert_eq!(StorageType::F16.decode(&0x3c00u16.to_le_bytes()), 1.0);
        assert_eq!(StorageType::Bf16.decode(&0xc000u16.to_le_bytes()), -2.0);
        assert_eq!(StorageType::I64.decode(&(-3i64).to_le_bytes()), -3.0);
    }

    #[test]
    fn tolerates_non_string_dict_keys() {
        let pickle = b"\x80\x02}q\x00(X\x01\x00\x00\x00aK\x01K\x02K\x03\x86q\x01K\x04uK\x05K\x06s.";
        let root = Machine::default().run(pickle).unwrap();
        let Value::Dict(items) = root else {
            panic!("expected dict")
        };
        assert_eq!(items.len(), 3);
        assert!(matches!(lookup(&Value::Dict(items), "a"), Some(Value::Int(1))));
    }

    #[test]
    fn strided_gather() {
        let bytes: Vec<u8> = (0..6).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let view = TensorRef {
            key: String::new(),
            dtype: StorageType::F32,
            offset: 0,
            shape: vec![3, 2],
            stride: vec![1, 3],
        };
        assert_eq!(gather(&bytes, &view).unwrap(), vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }
}
