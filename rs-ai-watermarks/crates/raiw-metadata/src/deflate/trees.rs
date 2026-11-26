//! Huffman tree construction and block emission (`trees.c`, shared by zlib and zlib-ng).

use super::bits::BitWriter;
use super::tables::{
    BL_CODES, BL_ORDER, Code, D_CODES, END_BLOCK, EXTRA_BLBITS, EXTRA_DBITS, EXTRA_LBITS, HEAP_SIZE, L_CODES,
    LITERALS, MAX_BITS, MAX_BL_BITS, REP_3_6, REPZ_3_10, REPZ_11_138, TABLES, bi_reverse, d_code,
};

const STORED_BLOCK: u64 = 0;
const STATIC_TREES: u64 = 1;
const DYN_TREES: u64 = 2;

/// Deflate strategies that change block or match decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    Default,
    Filtered,
    Fixed,
}

struct StaticDesc {
    stree: Option<&'static [Code]>,
    extra: &'static [u8],
    extra_base: usize,
    elems: usize,
    max_length: u32,
}

static L_DESC: StaticDesc = StaticDesc {
    stree: Some(&TABLES.static_ltree),
    extra: &EXTRA_LBITS,
    extra_base: LITERALS + 1,
    elems: L_CODES,
    max_length: MAX_BITS as u32,
};
static D_DESC: StaticDesc = StaticDesc {
    stree: Some(&TABLES.static_dtree),
    extra: &EXTRA_DBITS,
    extra_base: 0,
    elems: D_CODES,
    max_length: MAX_BITS as u32,
};
static BL_DESC: StaticDesc = StaticDesc {
    stree: None,
    extra: &EXTRA_BLBITS,
    extra_base: 0,
    elems: BL_CODES,
    max_length: MAX_BL_BITS as u32,
};

/// Heap, depth and bit-length scratch of `build_tree`.
#[derive(Debug)]
struct Builder {
    heap: [usize; 2 * L_CODES + 1],
    heap_len: usize,
    heap_max: usize,
    depth: [u8; 2 * L_CODES + 1],
    bl_count: [u16; MAX_BITS + 1],
    opt_len: u64,
    static_len: u64,
}

impl Builder {
    fn smaller(&self, tree: &[Code], n: usize, m: usize) -> bool {
        tree[n].freq_code < tree[m].freq_code
            || (tree[n].freq_code == tree[m].freq_code && self.depth[n] <= self.depth[m])
    }

    fn pqdownheap(&mut self, tree: &[Code], mut k: usize) {
        let v = self.heap[k];
        let mut j = k << 1;
        while j <= self.heap_len {
            if j < self.heap_len && self.smaller(tree, self.heap[j + 1], self.heap[j]) {
                j += 1;
            }
            if self.smaller(tree, v, self.heap[j]) {
                break;
            }
            self.heap[k] = self.heap[j];
            k = j;
            j <<= 1;
        }
        self.heap[k] = v;
    }

    /// Build an optimal tree for `tree`'s frequencies; returns `max_code`.
    fn build_tree(&mut self, tree: &mut [Code], desc: &StaticDesc) -> i32 {
        let elems = desc.elems;
        let mut max_code: i32 = -1;
        self.heap_len = 0;
        self.heap_max = HEAP_SIZE;
        for (n, code) in tree[..elems].iter_mut().enumerate() {
            if code.freq_code != 0 {
                self.heap_len += 1;
                self.heap[self.heap_len] = n;
                max_code = n as i32;
                self.depth[n] = 0;
            } else {
                code.dad_len = 0;
            }
        }
        while self.heap_len < 2 {
            let node = if max_code < 2 {
                max_code += 1;
                max_code as usize
            } else {
                0
            };
            self.heap_len += 1;
            self.heap[self.heap_len] = node;
            tree[node].freq_code = 1;
            self.depth[node] = 0;
            self.opt_len = self.opt_len.wrapping_sub(1);
            if let Some(stree) = desc.stree {
                self.static_len = self.static_len.wrapping_sub(u64::from(stree[node].dad_len));
            }
        }
        for n in (1..=self.heap_len / 2).rev() {
            self.pqdownheap(tree, n);
        }
        let mut node = elems;
        loop {
            let n = self.heap[1];
            self.heap[1] = self.heap[self.heap_len];
            self.heap_len -= 1;
            self.pqdownheap(tree, 1);
            let m = self.heap[1];
            self.heap_max -= 1;
            self.heap[self.heap_max] = n;
            self.heap_max -= 1;
            self.heap[self.heap_max] = m;
            tree[node].freq_code = tree[n].freq_code.wrapping_add(tree[m].freq_code);
            self.depth[node] = self.depth[n].max(self.depth[m]) + 1;
            tree[n].dad_len = node as u16;
            tree[m].dad_len = node as u16;
            self.heap[1] = node;
            node += 1;
            self.pqdownheap(tree, 1);
            if self.heap_len < 2 {
                break;
            }
        }
        self.heap_max -= 1;
        self.heap[self.heap_max] = self.heap[1];
        self.gen_bitlen(tree, max_code, desc);
        gen_codes(tree, max_code, &self.bl_count);
        max_code
    }

    fn gen_bitlen(&mut self, tree: &mut [Code], max_code: i32, desc: &StaticDesc) {
        let max_length = desc.max_length;
        let mut overflow: i32 = 0;
        self.bl_count = [0; MAX_BITS + 1];
        tree[self.heap[self.heap_max]].dad_len = 0;
        let mut h = self.heap_max + 1;
        while h < HEAP_SIZE {
            let n = self.heap[h];
            let mut bits = u32::from(tree[tree[n].dad_len as usize].dad_len) + 1;
            if bits > max_length {
                bits = max_length;
                overflow += 1;
            }
            tree[n].dad_len = bits as u16;
            h += 1;
            if n as i32 > max_code {
                continue;
            }
            self.bl_count[bits as usize] += 1;
            let xbits = if n >= desc.extra_base {
                u32::from(desc.extra[n - desc.extra_base])
            } else {
                0
            };
            let f = u64::from(tree[n].freq_code);
            self.opt_len = self.opt_len.wrapping_add(f * u64::from(bits + xbits));
            if let Some(stree) = desc.stree {
                self.static_len = self
                    .static_len
                    .wrapping_add(f * u64::from(u32::from(stree[n].dad_len) + xbits));
            }
        }
        if overflow == 0 {
            return;
        }
        loop {
            let mut bits = max_length as usize - 1;
            while self.bl_count[bits] == 0 {
                bits -= 1;
            }
            self.bl_count[bits] -= 1;
            self.bl_count[bits + 1] += 2;
            self.bl_count[max_length as usize] -= 1;
            overflow -= 2;
            if overflow <= 0 {
                break;
            }
        }
        let mut h = HEAP_SIZE;
        for bits in (1..=max_length as usize).rev() {
            let mut n = self.bl_count[bits];
            while n != 0 {
                h -= 1;
                let m = self.heap[h];
                if m as i32 > max_code {
                    continue;
                }
                if usize::from(tree[m].dad_len) != bits {
                    let freq = u64::from(tree[m].freq_code);
                    self.opt_len = self
                        .opt_len
                        .wrapping_add((bits as u64).wrapping_mul(freq))
                        .wrapping_sub(u64::from(tree[m].dad_len).wrapping_mul(freq));
                    tree[m].dad_len = bits as u16;
                }
                n -= 1;
            }
        }
    }
}

fn gen_codes(tree: &mut [Code], max_code: i32, bl_count: &[u16; MAX_BITS + 1]) {
    let mut next_code = [0u16; MAX_BITS + 1];
    let mut code = 0u32;
    for bits in 1..=MAX_BITS {
        code = (code + u32::from(bl_count[bits - 1])) << 1;
        next_code[bits] = code as u16;
    }
    let Ok(max_code) = usize::try_from(max_code) else {
        return;
    };
    for code in &mut tree[..=max_code] {
        let len = code.dad_len as usize;
        if len == 0 {
            continue;
        }
        code.freq_code = bi_reverse(u32::from(next_code[len]), len as u32);
        next_code[len] = next_code[len].wrapping_add(1);
    }
}

/// Block state: dynamic trees, the symbol buffer and the pending bit stream.
#[derive(Debug)]
pub struct Trees {
    dyn_ltree: [Code; HEAP_SIZE],
    dyn_dtree: [Code; 2 * D_CODES + 1],
    bl_tree: [Code; 2 * BL_CODES + 1],
    l_max_code: i32,
    d_max_code: i32,
    builder: Builder,
    d_buf: Vec<u16>,
    l_buf: Vec<u8>,
    sym_next: usize,
    sym_end: usize,
    pub matches: u32,
    strategy: Strategy,
    /// zlib-ng emits an empty static block without building trees.
    empty_shortcut: bool,
    pub out: BitWriter,
}

impl Trees {
    pub fn new(lit_bufsize: usize, strategy: Strategy, empty_shortcut: bool) -> Self {
        let mut trees = Self {
            dyn_ltree: [Code::default(); HEAP_SIZE],
            dyn_dtree: [Code::default(); 2 * D_CODES + 1],
            bl_tree: [Code::default(); 2 * BL_CODES + 1],
            l_max_code: 0,
            d_max_code: 0,
            builder: Builder {
                heap: [0; 2 * L_CODES + 1],
                heap_len: 0,
                heap_max: 0,
                depth: [0; 2 * L_CODES + 1],
                bl_count: [0; MAX_BITS + 1],
                opt_len: 0,
                static_len: 0,
            },
            d_buf: vec![0; lit_bufsize],
            l_buf: vec![0; lit_bufsize],
            sym_next: 0,
            sym_end: lit_bufsize - 1,
            matches: 0,
            strategy,
            empty_shortcut,
            out: BitWriter::default(),
        };
        trees.init_block();
        trees
    }

    fn init_block(&mut self) {
        for code in &mut self.dyn_ltree[..L_CODES] {
            code.freq_code = 0;
        }
        for code in &mut self.dyn_dtree[..D_CODES] {
            code.freq_code = 0;
        }
        for code in &mut self.bl_tree[..BL_CODES] {
            code.freq_code = 0;
        }
        self.dyn_ltree[END_BLOCK].freq_code = 1;
        self.builder.opt_len = 0;
        self.builder.static_len = 0;
        self.sym_next = 0;
        self.matches = 0;
    }

    pub fn sym_next(&self) -> usize {
        self.sym_next
    }

    /// Record a literal; true when the block must be flushed.
    #[inline]
    pub fn tally_lit(&mut self, c: u8) -> bool {
        self.d_buf[self.sym_next] = 0;
        self.l_buf[self.sym_next] = c;
        self.sym_next += 1;
        self.dyn_ltree[c as usize].freq_code = self.dyn_ltree[c as usize].freq_code.wrapping_add(1);
        self.sym_next == self.sym_end
    }

    /// Record a match of `dist` bytes back and length `len + MIN_MATCH`.
    #[inline]
    pub fn tally_dist(&mut self, dist: usize, len: usize) -> bool {
        self.d_buf[self.sym_next] = dist as u16;
        self.l_buf[self.sym_next] = len as u8;
        self.sym_next += 1;
        self.matches += 1;
        let lcode = TABLES.length_code[len] as usize + LITERALS + 1;
        self.dyn_ltree[lcode].freq_code = self.dyn_ltree[lcode].freq_code.wrapping_add(1);
        let dcode = d_code(dist - 1);
        self.dyn_dtree[dcode].freq_code = self.dyn_dtree[dcode].freq_code.wrapping_add(1);
        self.sym_next == self.sym_end
    }

    fn send_code(out: &mut BitWriter, tree: &[Code], c: usize) {
        out.send_bits(u64::from(tree[c].freq_code), u32::from(tree[c].dad_len));
    }

    fn scan_tree(bl_tree: &mut [Code], tree: &mut [Code], max_code: i32) {
        let mut prevlen: i32 = -1;
        let mut nextlen = i32::from(tree[0].dad_len);
        let mut count = 0u16;
        let (mut max_count, mut min_count) = if nextlen == 0 { (138u16, 3u16) } else { (7, 4) };
        let Ok(max_code) = usize::try_from(max_code) else {
            return;
        };
        tree[max_code + 1].dad_len = 0xffff;
        for n in 0..=max_code {
            let curlen = nextlen;
            nextlen = i32::from(tree[n + 1].dad_len);
            count += 1;
            if count < max_count && curlen == nextlen {
                continue;
            }
            if count < min_count {
                let slot = &mut bl_tree[curlen as usize].freq_code;
                *slot = slot.wrapping_add(count);
            } else if curlen != 0 {
                if curlen != prevlen {
                    bl_tree[curlen as usize].freq_code += 1;
                }
                bl_tree[REP_3_6].freq_code += 1;
            } else if count <= 10 {
                bl_tree[REPZ_3_10].freq_code += 1;
            } else {
                bl_tree[REPZ_11_138].freq_code += 1;
            }
            count = 0;
            prevlen = curlen;
            (max_count, min_count) = if nextlen == 0 {
                (138, 3)
            } else if curlen == nextlen {
                (6, 3)
            } else {
                (7, 4)
            };
        }
    }

    fn send_tree(out: &mut BitWriter, bl_tree: &[Code], tree: &[Code], max_code: i32) {
        let mut prevlen: i32 = -1;
        let mut nextlen = i32::from(tree[0].dad_len);
        let mut count: i32 = 0;
        let (mut max_count, mut min_count) = if nextlen == 0 { (138, 3) } else { (7, 4) };
        let Ok(max_code) = usize::try_from(max_code) else {
            return;
        };
        for n in 0..=max_code {
            let curlen = nextlen;
            nextlen = i32::from(tree[n + 1].dad_len);
            count += 1;
            if count < max_count && curlen == nextlen {
                continue;
            }
            if count < min_count {
                while count != 0 {
                    Self::send_code(out, bl_tree, curlen as usize);
                    count -= 1;
                }
            } else if curlen != 0 {
                if curlen != prevlen {
                    Self::send_code(out, bl_tree, curlen as usize);
                    count -= 1;
                }
                Self::send_code(out, bl_tree, REP_3_6);
                out.send_bits((count - 3) as u64, 2);
            } else if count <= 10 {
                Self::send_code(out, bl_tree, REPZ_3_10);
                out.send_bits((count - 3) as u64, 3);
            } else {
                Self::send_code(out, bl_tree, REPZ_11_138);
                out.send_bits((count - 11) as u64, 7);
            }
            count = 0;
            prevlen = curlen;
            (max_count, min_count) = if nextlen == 0 {
                (138, 3)
            } else if curlen == nextlen {
                (6, 3)
            } else {
                (7, 4)
            };
        }
    }

    fn build_bl_tree(&mut self) -> usize {
        Self::scan_tree(&mut self.bl_tree, &mut self.dyn_ltree, self.l_max_code);
        Self::scan_tree(&mut self.bl_tree, &mut self.dyn_dtree, self.d_max_code);
        self.builder.build_tree(&mut self.bl_tree, &BL_DESC);
        let mut max_blindex = BL_CODES - 1;
        while max_blindex >= 3 {
            if self.bl_tree[BL_ORDER[max_blindex] as usize].dad_len != 0 {
                break;
            }
            max_blindex -= 1;
        }
        self.builder.opt_len = self
            .builder
            .opt_len
            .wrapping_add(3 * (max_blindex as u64 + 1) + 5 + 5 + 4);
        max_blindex
    }

    fn send_all_trees(&mut self, lcodes: usize, dcodes: usize, blcodes: usize) {
        let out = &mut self.out;
        out.send_bits((lcodes - 257) as u64, 5);
        out.send_bits((dcodes - 1) as u64, 5);
        out.send_bits((blcodes - 4) as u64, 4);
        for &rank in &BL_ORDER[..blcodes] {
            out.send_bits(u64::from(self.bl_tree[rank as usize].dad_len), 3);
        }
        Self::send_tree(out, &self.bl_tree, &self.dyn_ltree, lcodes as i32 - 1);
        Self::send_tree(out, &self.bl_tree, &self.dyn_dtree, dcodes as i32 - 1);
    }

    fn compress_block(&mut self, dynamic: bool) {
        let (ltree, dtree): (&[Code], &[Code]) = if dynamic {
            (&self.dyn_ltree, &self.dyn_dtree)
        } else {
            (&TABLES.static_ltree, &TABLES.static_dtree)
        };
        let out = &mut self.out;
        for sx in 0..self.sym_next {
            let dist = self.d_buf[sx] as usize;
            let lc = self.l_buf[sx] as usize;
            if dist == 0 {
                Self::send_code(out, ltree, lc);
            } else {
                let code = TABLES.length_code[lc] as usize;
                let c = code + LITERALS + 1;
                let mut bits = u64::from(ltree[c].freq_code);
                let mut nbits = u32::from(ltree[c].dad_len);
                let extra = u32::from(EXTRA_LBITS[code]);
                if extra != 0 {
                    bits |= ((lc - TABLES.base_length[code] as usize) as u64) << nbits;
                    nbits += extra;
                }
                let dist = dist - 1;
                let code = d_code(dist);
                bits |= u64::from(dtree[code].freq_code) << nbits;
                nbits += u32::from(dtree[code].dad_len);
                let extra = u32::from(EXTRA_DBITS[code]);
                if extra != 0 {
                    bits |= ((dist - TABLES.base_dist[code] as usize) as u64) << nbits;
                    nbits += extra;
                }
                out.send_bits(bits, nbits);
            }
        }
        Self::send_code(out, ltree, END_BLOCK);
    }

    /// `_tr_stored_block`.
    pub fn stored_block(&mut self, buf: &[u8], last: bool) {
        let stored_len = buf.len();
        self.out.send_bits((STORED_BLOCK << 1) + u64::from(last), 3);
        self.out.windup();
        self.out.put_u16_le(stored_len as u16);
        self.out.put_u16_le(!(stored_len as u16));
        self.out.put_bytes(buf);
    }

    /// `_tr_flush_block`: `buf` is the block's window bytes when they are still there.
    pub fn flush_block(&mut self, buf: Option<&[u8]>, stored_len: usize, last: bool) {
        let mut max_blindex = 0;
        let (mut opt_lenb, static_lenb);
        if self.empty_shortcut && self.sym_next == 0 {
            opt_lenb = 0;
            static_lenb = 0;
            self.builder.static_len = 7;
        } else {
            self.l_max_code = self.builder.build_tree(&mut self.dyn_ltree, &L_DESC);
            self.d_max_code = self.builder.build_tree(&mut self.dyn_dtree, &D_DESC);
            max_blindex = self.build_bl_tree();
            opt_lenb = (self.builder.opt_len.wrapping_add(3 + 7)) >> 3;
            static_lenb = (self.builder.static_len.wrapping_add(3 + 7)) >> 3;
            if static_lenb <= opt_lenb || self.strategy == Strategy::Fixed {
                opt_lenb = static_lenb;
            }
        }
        match buf {
            Some(data) if stored_len as u64 + 4 <= opt_lenb => {
                self.stored_block(&data[..stored_len], last);
            }
            _ if static_lenb == opt_lenb => {
                self.out.send_bits((STATIC_TREES << 1) + u64::from(last), 3);
                self.compress_block(false);
            }
            _ => {
                self.out.send_bits((DYN_TREES << 1) + u64::from(last), 3);
                let lcodes = (self.l_max_code + 1) as usize;
                let dcodes = (self.d_max_code + 1) as usize;
                self.send_all_trees(lcodes, dcodes, max_blindex + 1);
                self.compress_block(true);
            }
        }
        self.init_block();
        if last {
            self.out.windup();
        }
    }
}
