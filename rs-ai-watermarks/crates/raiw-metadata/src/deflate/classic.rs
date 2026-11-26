//! madler zlib 1.4 `deflate` with `deflate_slow` (levels 4 to 9).
//!
//! This is the `libz.so.1` the reference interpreter and Pillow's `_imaging` both load,
//! so it produces Python's `zlib.compress` output and Pillow's PNG `IDAT` stream.

use super::bits::adler32;
use super::tables::{MAX_MATCH, MIN_LOOKAHEAD, MIN_MATCH};
use super::trees::{Strategy, Trees};
use super::{DeflateError, Flush, Input, Output, Status};

const TOO_FAR: usize = 4096;
const WINDOW_BITS: u32 = 15;

/// `(good, lazy, nice, chain)` of zlib's lazy levels 4 to 9.
const CONFIG: [(usize, usize, usize, usize); 6] = [
    (4, 4, 16, 16),
    (8, 16, 32, 32),
    (8, 16, 128, 128),
    (8, 32, 128, 256),
    (32, 128, 258, 1024),
    (32, 258, 258, 4096),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockState {
    NeedMore,
    BlockDone,
    FinishStarted,
    FinishDone,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Init,
    Busy,
    Finish,
}

/// A zlib deflate stream (zlib wrapper, 32 KiB window).
#[derive(Debug)]
pub struct Deflater {
    window: Vec<u8>,
    prev: Vec<u16>,
    head: Vec<u16>,
    w_size: usize,
    w_mask: usize,
    window_size: usize,
    hash_mask: usize,
    hash_shift: u32,
    ins_h: usize,
    high_water: usize,
    lookahead: usize,
    strstart: usize,
    block_start: isize,
    match_start: usize,
    match_length: usize,
    prev_length: usize,
    prev_match: usize,
    match_available: bool,
    insert: usize,
    level: u8,
    good_match: usize,
    max_lazy: usize,
    nice_match: usize,
    max_chain: usize,
    strategy: Strategy,
    phase: Phase,
    last_flush: i32,
    trailer_written: bool,
    adler: u32,
    trees: Trees,
    pending_out: usize,
}

const fn rank(flush: Flush) -> i32 {
    match flush {
        Flush::None => 0,
        Flush::Finish => 8,
    }
}

impl Deflater {
    /// `deflateInit2(level, Z_DEFLATED, 15, mem_level, strategy)` for levels 4 to 9.
    pub fn new(level: u8, mem_level: u8, strategy: Strategy) -> Result<Self, DeflateError> {
        if !(4..=9).contains(&level) {
            return Err(DeflateError::Level(level));
        }
        if !(1..=9).contains(&mem_level) {
            return Err(DeflateError::MemLevel(mem_level));
        }
        let w_size = 1usize << WINDOW_BITS;
        let hash_bits = u32::from(mem_level) + 7;
        let (good, lazy, nice, chain) = CONFIG[usize::from(level) - 4];
        Ok(Self {
            window: vec![0; 2 * w_size + 8],
            prev: vec![0; w_size],
            head: vec![0; 1 << hash_bits],
            w_size,
            w_mask: w_size - 1,
            window_size: 2 * w_size,
            hash_mask: (1 << hash_bits) - 1,
            hash_shift: hash_bits.div_ceil(MIN_MATCH as u32),
            ins_h: 0,
            high_water: 0,
            lookahead: 0,
            strstart: 0,
            block_start: 0,
            match_start: 0,
            match_length: MIN_MATCH - 1,
            prev_length: MIN_MATCH - 1,
            prev_match: 0,
            match_available: false,
            insert: 0,
            level,
            good_match: good,
            max_lazy: lazy,
            nice_match: nice,
            max_chain: chain,
            strategy,
            phase: Phase::Init,
            last_flush: -2,
            trailer_written: false,
            adler: 1,
            trees: Trees::new(1 << (mem_level + 6), strategy, false),
            pending_out: 0,
        })
    }

    fn max_dist(&self) -> usize {
        self.w_size - MIN_LOOKAHEAD
    }

    fn pending(&self) -> usize {
        self.trees.out.pending.len() - self.pending_out
    }

    #[inline]
    fn update_hash(&self, h: usize, c: u8) -> usize {
        ((h << self.hash_shift) ^ usize::from(c)) & self.hash_mask
    }

    /// `INSERT_STRING`: returns the previous chain head.
    #[inline]
    fn insert_string(&mut self, str: usize) -> usize {
        self.ins_h = self.update_hash(self.ins_h, self.window[str + MIN_MATCH - 1]);
        let head = self.head[self.ins_h];
        self.prev[str & self.w_mask] = head;
        self.head[self.ins_h] = str as u16;
        usize::from(head)
    }

    fn slide_hash(&mut self) {
        let wsize = self.w_size as u16;
        for entry in self.head.iter_mut().chain(self.prev.iter_mut()) {
            *entry = entry.saturating_sub(wsize);
        }
    }

    fn fill_window(&mut self, input: &mut Input<'_>) {
        let wsize = self.w_size;
        loop {
            let mut more = self.window_size - self.lookahead - self.strstart;
            if self.strstart >= wsize + self.max_dist() {
                self.window.copy_within(wsize..2 * wsize - more, 0);
                self.match_start = self.match_start.wrapping_sub(wsize);
                self.strstart -= wsize;
                self.block_start -= wsize as isize;
                if self.insert > self.strstart {
                    self.insert = self.strstart;
                }
                self.slide_hash();
                more += wsize;
            }
            if input.remaining() == 0 {
                break;
            }
            let at = self.strstart + self.lookahead;
            let taken = input.take(more);
            self.window[at..at + taken.len()].copy_from_slice(taken);
            self.adler = adler32(self.adler, taken);
            self.lookahead += taken.len();
            if self.lookahead + self.insert >= MIN_MATCH {
                let mut str = self.strstart - self.insert;
                self.ins_h = usize::from(self.window[str]);
                self.ins_h = self.update_hash(self.ins_h, self.window[str + 1]);
                while self.insert != 0 {
                    self.ins_h = self.update_hash(self.ins_h, self.window[str + MIN_MATCH - 1]);
                    self.prev[str & self.w_mask] = self.head[self.ins_h];
                    self.head[self.ins_h] = str as u16;
                    str += 1;
                    self.insert -= 1;
                    if self.lookahead + self.insert < MIN_MATCH {
                        break;
                    }
                }
            }
            if !(self.lookahead < MIN_LOOKAHEAD && input.remaining() != 0) {
                break;
            }
        }
        if self.high_water < self.window_size {
            let curr = self.strstart + self.lookahead;
            if self.high_water < curr {
                let init = (self.window_size - curr).min(MAX_MATCH);
                self.window[curr..curr + init].fill(0);
                self.high_water = curr + init;
            } else if self.high_water < curr + MAX_MATCH {
                let init = (curr + MAX_MATCH - self.high_water).min(self.window_size - self.high_water);
                self.window[self.high_water..self.high_water + init].fill(0);
                self.high_water += init;
            }
        }
    }

    fn longest_match(&mut self, mut cur_match: usize) -> usize {
        let mut chain_length = self.max_chain;
        let scan = self.strstart;
        let mut best_len = self.prev_length;
        let mut nice_match = self.nice_match;
        let limit = if self.strstart > self.max_dist() {
            self.strstart - self.max_dist()
        } else {
            0
        };
        let w = &self.window;
        let mut scan_end1 = w[scan + best_len - 1];
        let mut scan_end = w[scan + best_len];
        if self.prev_length >= self.good_match {
            chain_length >>= 2;
        }
        if nice_match > self.lookahead {
            nice_match = self.lookahead;
        }
        loop {
            let m = cur_match;
            if w[m + best_len] == scan_end
                && w[m + best_len - 1] == scan_end1
                && w[m] == w[scan]
                && w[m + 1] == w[scan + 1]
            {
                // Byte 2 is implied by equal hashes and is never compared.
                let len = 3 + w[scan + 3..scan + MAX_MATCH]
                    .iter()
                    .zip(&w[m + 3..m + MAX_MATCH])
                    .take_while(|(a, b)| a == b)
                    .count();
                if len > best_len {
                    self.match_start = cur_match;
                    best_len = len;
                    if len >= nice_match {
                        break;
                    }
                    scan_end1 = w[scan + best_len - 1];
                    scan_end = w[scan + best_len];
                }
            }
            cur_match = usize::from(self.prev[cur_match & self.w_mask]);
            if cur_match <= limit {
                break;
            }
            chain_length -= 1;
            if chain_length == 0 {
                break;
            }
        }
        best_len.min(self.lookahead)
    }

    fn flush_block_only(&mut self, last: bool, out: &mut Output) {
        let stored_len = (self.strstart as isize - self.block_start) as usize;
        let buf = usize::try_from(self.block_start)
            .ok()
            .map(|start| &self.window[start..]);
        self.trees.flush_block(buf, stored_len, last);
        self.block_start = self.strstart as isize;
        self.flush_pending(out);
    }

    fn deflate_slow(&mut self, input: &mut Input<'_>, out: &mut Output, flush: Flush) -> BlockState {
        loop {
            if self.lookahead < MIN_LOOKAHEAD {
                self.fill_window(input);
                if self.lookahead < MIN_LOOKAHEAD && flush == Flush::None {
                    return BlockState::NeedMore;
                }
                if self.lookahead == 0 {
                    break;
                }
            }
            let mut hash_head = 0;
            if self.lookahead >= MIN_MATCH {
                hash_head = self.insert_string(self.strstart);
            }
            self.prev_length = self.match_length;
            self.prev_match = self.match_start;
            self.match_length = MIN_MATCH - 1;
            if hash_head != 0
                && self.prev_length < self.max_lazy
                && self.strstart.wrapping_sub(hash_head) <= self.max_dist()
            {
                self.match_length = self.longest_match(hash_head);
                if self.match_length <= 5
                    && (self.strategy == Strategy::Filtered
                        || (self.match_length == MIN_MATCH
                            && self.strstart.wrapping_sub(self.match_start) > TOO_FAR))
                {
                    self.match_length = MIN_MATCH - 1;
                }
            }
            if self.prev_length >= MIN_MATCH && self.match_length <= self.prev_length {
                let max_insert = self.strstart + self.lookahead - MIN_MATCH;
                let bflush = self
                    .trees
                    .tally_dist(self.strstart - 1 - self.prev_match, self.prev_length - MIN_MATCH);
                self.lookahead -= self.prev_length - 1;
                self.prev_length -= 2;
                loop {
                    self.strstart += 1;
                    if self.strstart <= max_insert {
                        self.insert_string(self.strstart);
                    }
                    self.prev_length -= 1;
                    if self.prev_length == 0 {
                        break;
                    }
                }
                self.match_available = false;
                self.match_length = MIN_MATCH - 1;
                self.strstart += 1;
                if bflush {
                    self.flush_block_only(false, out);
                    if out.avail == 0 {
                        return BlockState::NeedMore;
                    }
                }
            } else if self.match_available {
                let bflush = self.trees.tally_lit(self.window[self.strstart - 1]);
                if bflush {
                    self.flush_block_only(false, out);
                }
                self.strstart += 1;
                self.lookahead -= 1;
                if out.avail == 0 {
                    return BlockState::NeedMore;
                }
            } else {
                self.match_available = true;
                self.strstart += 1;
                self.lookahead -= 1;
            }
        }
        if self.match_available {
            self.trees.tally_lit(self.window[self.strstart - 1]);
            self.match_available = false;
        }
        self.insert = self.strstart.min(MIN_MATCH - 1);
        if flush == Flush::Finish {
            self.flush_block_only(true, out);
            if out.avail == 0 {
                return BlockState::FinishStarted;
            }
            return BlockState::FinishDone;
        }
        if self.trees.sym_next() != 0 {
            self.flush_block_only(false, out);
            if out.avail == 0 {
                return BlockState::NeedMore;
            }
        }
        BlockState::BlockDone
    }

    fn flush_pending(&mut self, out: &mut Output) {
        self.trees.out.flush_bits();
        let len = self.pending().min(out.avail);
        if len == 0 {
            return;
        }
        let start = self.pending_out;
        out.data
            .extend_from_slice(&self.trees.out.pending[start..start + len]);
        out.avail -= len;
        self.pending_out += len;
        if self.pending() == 0 {
            self.trees.out.pending.clear();
            self.pending_out = 0;
        }
    }

    /// One `deflate(strm, flush)` call over `input`, writing at most `out.avail` bytes.
    pub fn deflate(
        &mut self,
        input: &mut Input<'_>,
        out: &mut Output,
        flush: Flush,
    ) -> Result<Status, DeflateError> {
        if out.avail == 0 {
            return Err(DeflateError::Buffer);
        }
        let old_flush = self.last_flush;
        self.last_flush = rank(flush);
        if self.pending() != 0 {
            self.flush_pending(out);
            if out.avail == 0 {
                self.last_flush = -1;
                return Ok(Status::Ok);
            }
        } else if input.remaining() == 0 && rank(flush) <= old_flush && flush != Flush::Finish {
            return Err(DeflateError::Buffer);
        }
        if self.phase == Phase::Finish && input.remaining() != 0 {
            return Err(DeflateError::Buffer);
        }
        if self.phase == Phase::Init {
            let mut header: u32 = (8 + ((WINDOW_BITS - 8) << 4)) << 8;
            let level_flags: u32 = match self.level {
                _ if self.strategy == Strategy::Fixed => 0,
                4 | 5 => 1,
                6 => 2,
                _ => 3,
            };
            header |= level_flags << 6;
            header += 31 - (header % 31);
            self.trees.out.put_u16_be(header as u16);
            self.phase = Phase::Busy;
            self.flush_pending(out);
            if self.pending() != 0 {
                self.last_flush = -1;
                return Ok(Status::Ok);
            }
        }
        if input.remaining() != 0
            || self.lookahead != 0
            || (flush != Flush::None && self.phase != Phase::Finish)
        {
            let bstate = self.deflate_slow(input, out, flush);
            if matches!(bstate, BlockState::FinishStarted | BlockState::FinishDone) {
                self.phase = Phase::Finish;
            }
            if matches!(bstate, BlockState::NeedMore | BlockState::FinishStarted) {
                if out.avail == 0 {
                    self.last_flush = -1;
                }
                return Ok(Status::Ok);
            }
            if bstate == BlockState::BlockDone {
                self.flush_pending(out);
                if out.avail == 0 {
                    self.last_flush = -1;
                    return Ok(Status::Ok);
                }
            }
        }
        if flush != Flush::Finish {
            return Ok(Status::Ok);
        }
        if !self.trailer_written {
            self.trees.out.put_u32_be(self.adler);
            self.trailer_written = true;
        }
        self.flush_pending(out);
        if self.pending() == 0 {
            return Ok(Status::StreamEnd);
        }
        Ok(Status::Ok)
    }
}

/// `zlib.compress(data, level)` for the lazy-matching levels 4 to 9.
pub fn zlib_compress_level(data: &[u8], level: u8) -> Result<Vec<u8>, DeflateError> {
    let mut deflater = Deflater::new(level, 8, Strategy::Default)?;
    let mut input = Input::new(data);
    let mut out = Output {
        data: Vec::with_capacity(data.len() / 2 + 64),
        avail: usize::MAX,
    };
    loop {
        if deflater.deflate(&mut input, &mut out, Flush::Finish)? == Status::StreamEnd {
            return Ok(out.data);
        }
    }
}

/// Python's `zlib.compress(data)`: level 6, 32 KiB window, memLevel 8, default strategy.
pub fn zlib_compress(data: &[u8]) -> Vec<u8> {
    zlib_compress_level(data, 6).unwrap_or_default()
}
