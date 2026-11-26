//! The fixed deflate tables (RFC 1951), built at compile time as zlib's `tr_static_init` does.

pub const LENGTH_CODES: usize = 29;
pub const LITERALS: usize = 256;
pub const L_CODES: usize = LITERALS + 1 + LENGTH_CODES;
pub const D_CODES: usize = 30;
pub const BL_CODES: usize = 19;
pub const HEAP_SIZE: usize = 2 * L_CODES + 1;
pub const MAX_BITS: usize = 15;
pub const MAX_BL_BITS: usize = 7;
pub const END_BLOCK: usize = 256;
pub const REP_3_6: usize = 16;
pub const REPZ_3_10: usize = 17;
pub const REPZ_11_138: usize = 18;

pub const MIN_MATCH: usize = 3;
pub const MAX_MATCH: usize = 258;
pub const MIN_LOOKAHEAD: usize = MAX_MATCH + MIN_MATCH + 1;

pub const EXTRA_LBITS: [u8; LENGTH_CODES] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
pub const EXTRA_DBITS: [u8; D_CODES] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
];
pub const EXTRA_BLBITS: [u8; BL_CODES] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 7];
pub const BL_ORDER: [u8; BL_CODES] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

/// One Huffman code: `(code, len)` once built, `(freq, dad)` while building.
#[derive(Debug, Clone, Copy, Default)]
pub struct Code {
    pub freq_code: u16,
    pub dad_len: u16,
}

#[derive(Debug)]
pub struct Tables {
    pub length_code: [u8; MAX_MATCH - MIN_MATCH + 1],
    pub dist_code: [u8; 512],
    pub base_length: [u16; LENGTH_CODES],
    pub base_dist: [u16; D_CODES],
    pub static_ltree: [Code; L_CODES + 2],
    pub static_dtree: [Code; D_CODES],
}

/// Reverse the low `len` bits of `code`.
pub const fn bi_reverse(mut code: u32, len: u32) -> u16 {
    let mut res = 0u32;
    let mut i = 0;
    while i < len {
        res = (res << 1) | (code & 1);
        code >>= 1;
        i += 1;
    }
    res as u16
}

const fn build() -> Tables {
    let mut t = Tables {
        length_code: [0; MAX_MATCH - MIN_MATCH + 1],
        dist_code: [0; 512],
        base_length: [0; LENGTH_CODES],
        base_dist: [0; D_CODES],
        static_ltree: [Code {
            freq_code: 0,
            dad_len: 0,
        }; L_CODES + 2],
        static_dtree: [Code {
            freq_code: 0,
            dad_len: 0,
        }; D_CODES],
    };
    // Length and distance code maps
    let mut length = 0usize;
    let mut code = 0usize;
    while code < LENGTH_CODES - 1 {
        t.base_length[code] = length as u16;
        let mut n = 0;
        while n < (1usize << EXTRA_LBITS[code]) {
            t.length_code[length] = code as u8;
            length += 1;
            n += 1;
        }
        code += 1;
    }
    t.length_code[length - 1] = code as u8;
    let mut dist = 0usize;
    code = 0;
    while code < 16 {
        t.base_dist[code] = dist as u16;
        let mut n = 0;
        while n < (1usize << EXTRA_DBITS[code]) {
            t.dist_code[dist] = code as u8;
            dist += 1;
            n += 1;
        }
        code += 1;
    }
    dist >>= 7;
    while code < D_CODES {
        t.base_dist[code] = (dist << 7) as u16;
        let mut n = 0;
        while n < (1usize << (EXTRA_DBITS[code] - 7)) {
            t.dist_code[256 + dist] = code as u8;
            dist += 1;
            n += 1;
        }
        code += 1;
    }
    // Static literal/length tree
    let mut bl_count = [0u16; MAX_BITS + 1];
    let mut n = 0usize;
    while n < L_CODES + 2 {
        let len = if n <= 143 {
            8
        } else if n <= 255 {
            9
        } else if n <= 279 {
            7
        } else {
            8
        };
        t.static_ltree[n].dad_len = len;
        bl_count[len as usize] += 1;
        n += 1;
    }
    let mut next_code = [0u16; MAX_BITS + 1];
    let mut acc = 0u32;
    let mut bits = 1;
    while bits <= MAX_BITS {
        acc = (acc + bl_count[bits - 1] as u32) << 1;
        next_code[bits] = acc as u16;
        bits += 1;
    }
    n = 0;
    while n <= L_CODES + 1 {
        let len = t.static_ltree[n].dad_len as usize;
        t.static_ltree[n].freq_code = bi_reverse(next_code[len] as u32, len as u32);
        next_code[len] += 1;
        n += 1;
    }
    // Static distance tree
    n = 0;
    while n < D_CODES {
        t.static_dtree[n].dad_len = 5;
        t.static_dtree[n].freq_code = bi_reverse(n as u32, 5);
        n += 1;
    }
    t
}

pub static TABLES: Tables = build();

/// Distance code for `dist` = match distance - 1.
#[inline]
pub fn d_code(dist: usize) -> usize {
    if dist < 256 {
        TABLES.dist_code[dist] as usize
    } else {
        TABLES.dist_code[256 + (dist >> 7)] as usize
    }
}
