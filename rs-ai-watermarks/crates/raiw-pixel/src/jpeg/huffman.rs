//! Baseline JFIF writer with the Annex K Huffman tables (`jcmarker.c`, `jchuff.c`, `jccoefct.c`).

use super::{Component, Encoded};

const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
    28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61,
    54, 47, 55, 62, 63,
];

const DC_LUMA_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const DC_CHROMA_BITS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
const DC_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const AC_LUMA_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];
const AC_LUMA_VALUES: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14,
    0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09,
    0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a,
    0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65,
    0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88,
    0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9,
    0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca,
    0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
    0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
];
const AC_CHROMA_BITS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
const AC_CHROMA_VALUES: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32,
    0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16,
    0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64,
    0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86,
    0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8,
    0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
    0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
];

/// Canonical code `(code, length)` per symbol (`jpeg_make_c_derived_tbl`).
struct Table {
    codes: [(u32, u8); 256],
}

impl Table {
    fn new(bits: &[u8; 16], values: &[u8]) -> Self {
        let mut codes = [(0, 0); 256];
        let (mut code, mut k) = (0_u32, 0);
        for (length, &count) in bits.iter().enumerate() {
            for _ in 0..count {
                codes[usize::from(values[k])] = (code, length as u8 + 1);
                code += 1;
                k += 1;
            }
            code <<= 1;
        }
        Self { codes }
    }
}

/// MSB-first bit sink with `0xFF` byte stuffing.
struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    count: u32,
}

impl BitWriter {
    fn put(&mut self, value: u32, length: u8) {
        if length == 0 {
            return;
        }
        let length = u32::from(length);
        self.acc = (self.acc << length) | u64::from(value & ((1 << length) - 1));
        self.count += length;
        while self.count >= 8 {
            let byte = (self.acc >> (self.count - 8)) as u8;
            self.out.push(byte);
            if byte == 0xFF {
                self.out.push(0);
            }
            self.count -= 8;
        }
    }

    fn flush(&mut self) {
        let pad = (8 - self.count % 8) % 8;
        self.put((1 << pad) - 1, pad as u8);
    }
}

fn magnitude(value: i32) -> (u32, u8) {
    let bits = 32 - value.unsigned_abs().leading_zeros();
    let raw = if value < 0 { value - 1 } else { value };
    (raw as u32, bits as u8)
}

fn encode_block(w: &mut BitWriter, block: &[i32; 64], last_dc: &mut i32, dc: &Table, ac: &Table) {
    let (bits, length) = magnitude(block[0] - *last_dc);
    *last_dc = block[0];
    let (code, size) = dc.codes[usize::from(length)];
    w.put(code, size);
    w.put(bits, length);
    let mut run = 0;
    for &k in &ZIGZAG[1..] {
        let value = block[k];
        if value == 0 {
            run += 1;
            continue;
        }
        while run > 15 {
            let (code, size) = ac.codes[0xF0];
            w.put(code, size);
            run -= 16;
        }
        let (bits, length) = magnitude(value);
        let (code, size) = ac.codes[(run << 4) + usize::from(length)];
        w.put(code, size);
        w.put(bits, length);
        run = 0;
    }
    if run > 0 {
        let (code, size) = ac.codes[0];
        w.put(code, size);
    }
}

fn marker(out: &mut Vec<u8>, code: u8, payload: &[u8]) {
    out.extend_from_slice(&[0xFF, code]);
    out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(payload);
}

fn dht(out: &mut Vec<u8>, class_id: u8, bits: &[u8; 16], values: &[u8]) {
    let mut payload = vec![class_id];
    payload.extend_from_slice(bits);
    payload.extend_from_slice(values);
    marker(out, 0xC4, &payload);
}

/// The MCU-ordered blocks of one component, with libjpeg's right and bottom dummy blocks.
fn mcu_blocks(c: &Component, mcu_w: usize, mcu_h: usize, mcu_x: usize, mcu_y: usize, out: &mut Vec<[i32; 64]>) {
    let start = out.len();
    for yi in 0..mcu_h {
        let by = mcu_y * mcu_h + yi;
        for xi in 0..mcu_w {
            let bx = mcu_x * mcu_w + xi;
            let block = if by < c.blocks_high && bx < c.blocks_wide {
                c.coefficients[by * c.blocks_wide + bx]
            } else {
                let mut dummy = [0; 64];
                let previous = if by < c.blocks_high { out.len() - 1 } else { start + yi * mcu_w - 1 };
                dummy[0] = out.get(previous).map_or(0, |b| b[0]);
                dummy
            };
            out.push(block);
        }
    }
}

/// Serialize the coefficients as Pillow/libjpeg-turbo writes them.
pub(super) fn write_jfif(encoded: &Encoded) -> Vec<u8> {
    let mut out = vec![0xFF, 0xD8];
    marker(&mut out, 0xE0, b"JFIF\0\x01\x01\x00\x00\x01\x00\x01\x00\x00");
    for (id, table) in encoded.tables.iter().enumerate() {
        let mut payload = vec![id as u8];
        payload.extend(ZIGZAG.iter().map(|&k| table[k] as u8));
        marker(&mut out, 0xDB, &payload);
    }
    let (w, h) = (encoded.width as u16, encoded.height as u16);
    let mut sof = vec![8];
    sof.extend_from_slice(&h.to_be_bytes());
    sof.extend_from_slice(&w.to_be_bytes());
    sof.extend_from_slice(&[3, 1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1]);
    marker(&mut out, 0xC0, &sof);
    dht(&mut out, 0x00, &DC_LUMA_BITS, &DC_VALUES);
    dht(&mut out, 0x10, &AC_LUMA_BITS, &AC_LUMA_VALUES);
    dht(&mut out, 0x01, &DC_CHROMA_BITS, &DC_VALUES);
    dht(&mut out, 0x11, &AC_CHROMA_BITS, &AC_CHROMA_VALUES);
    marker(&mut out, 0xDA, &[3, 1, 0x00, 2, 0x11, 3, 0x11, 0, 63, 0]);

    let tables = [
        (Table::new(&DC_LUMA_BITS, &DC_VALUES), Table::new(&AC_LUMA_BITS, &AC_LUMA_VALUES)),
        (Table::new(&DC_CHROMA_BITS, &DC_VALUES), Table::new(&AC_CHROMA_BITS, &AC_CHROMA_VALUES)),
    ];
    let mut writer = BitWriter { out, acc: 0, count: 0 };
    let mut last_dc = [0_i32; 3];
    let (mcus_wide, mcus_high) = (encoded.width.div_ceil(16), encoded.height.div_ceil(16));
    let mut blocks = Vec::with_capacity(6);
    for mcu_y in 0..mcus_high {
        for mcu_x in 0..mcus_wide {
            for (ci, component) in encoded.components.iter().enumerate() {
                let factor = if ci == 0 { 2 } else { 1 };
                blocks.clear();
                mcu_blocks(component, factor, factor, mcu_x, mcu_y, &mut blocks);
                let (dc, ac) = &tables[component.table];
                for block in &blocks {
                    encode_block(&mut writer, block, &mut last_dc[ci], dc, ac);
                }
            }
        }
    }
    writer.flush();
    let mut out = writer.out;
    out.extend_from_slice(&[0xFF, 0xD9]);
    out
}
