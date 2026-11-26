//! zlib-ng 2.3.3 `deflate` with the `deflate_medium` strategy (levels 3 to 6).
//!
//! The medium strategy keeps its look-ahead match in locals, so a call that returns early
//! (input exhausted, output full) changes the emitted stream: callers must reproduce the
//! exact input slices and output budget of the reference driver.

use super::bits::adler32;
use super::tables::{MAX_MATCH, MIN_LOOKAHEAD, MIN_MATCH};
use super::trees::{Strategy, Trees};
use super::{DeflateError, Flush, Input, Output, Status};

const WANT_MIN_MATCH: usize = 4;
const HASH_SIZE: usize = 65536;
const WINDOW_BITS: u32 = 15;

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

/// `(good, lazy, nice, chain)` of zlib-ng's medium levels.
const CONFIG: [(usize, usize, usize, usize); 4] =
    [(4, 6, 16, 6), (4, 12, 32, 24), (8, 16, 32, 32), (8, 16, 128, 128)];

#[derive(Debug, Clone, Copy, Default)]
struct Candidate {
    match_start: u16,
    match_length: u16,
    strstart: u16,
    orgstart: u16,
}

/// A zlib-ng deflate stream (zlib wrapper, 32 KiB window).
#[derive(Debug)]
pub struct Deflater {
    window: Vec<u8>,
    prev: Vec<u16>,
    head: Vec<u16>,
    w_size: usize,
    w_mask: usize,
    window_size: usize,
    high_water: usize,
    lookahead: usize,
    strstart: usize,
    block_start: isize,
    match_start: usize,
    prev_length: usize,
    insert: usize,
    level: usize,
    strategy: Strategy,
    max_chain: usize,
    max_lazy: usize,
    good_match: usize,
    nice_match: usize,
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
    /// `deflateInit2(level, Z_DEFLATED, 15, mem_level, strategy)` for levels 3 to 6.
    pub fn new(level: u8, mem_level: u8, strategy: Strategy) -> Result<Self, DeflateError> {
        if !(3..=6).contains(&level) {
            return Err(DeflateError::Level(level));
        }
        if !(1..=9).contains(&mem_level) {
            return Err(DeflateError::MemLevel(mem_level));
        }
        let w_size = 1usize << WINDOW_BITS;
        let lit_bufsize = 1usize << (mem_level + 6);
        let (good, lazy, nice, chain) = CONFIG[usize::from(level) - 3];
        Ok(Self {
            window: vec![0; 2 * w_size + 8],
            prev: vec![0; w_size],
            head: vec![0; HASH_SIZE],
            w_size,
            w_mask: w_size - 1,
            window_size: 2 * w_size,
            high_water: 0,
            lookahead: 0,
            strstart: 0,
            block_start: 0,
            match_start: 0,
            prev_length: 0,
            insert: 0,
            level: usize::from(level),
            strategy,
            max_chain: chain,
            max_lazy: lazy,
            good_match: good,
            nice_match: nice,
            phase: Phase::Init,
            last_flush: -2,
            trailer_written: false,
            adler: 1,
            trees: Trees::new(lit_bufsize, strategy, true),
            pending_out: 0,
        })
    }

    fn max_dist(&self) -> usize {
        self.w_size - MIN_LOOKAHEAD
    }

    fn pending(&self) -> usize {
        self.trees.out.pending.len() - self.pending_out
    }

    // ---- hashing ----

    #[inline]
    fn hash(&self, str: usize) -> usize {
        let val = u32::from_le_bytes([
            self.window[str],
            self.window[str + 1],
            self.window[str + 2],
            self.window[str + 3],
        ]);
        (val.wrapping_mul(2_654_435_761) >> 16) as usize & (HASH_SIZE - 1)
    }

    #[inline]
    fn quick_insert_string(&mut self, str: usize) -> usize {
        let hm = self.hash(str);
        let head = self.head[hm];
        if usize::from(head) != str {
            self.prev[str & self.w_mask] = head;
            self.head[hm] = str as u16;
        }
        usize::from(head)
    }

    fn insert_string(&mut self, str: usize, count: usize) {
        let mut idx = str as u16;
        for pos in str..str + count {
            let hm = self.hash(pos);
            let head = self.head[hm];
            if head != idx {
                self.prev[usize::from(idx) & self.w_mask] = head;
                self.head[hm] = idx;
            }
            idx = idx.wrapping_add(1);
        }
    }

    // ---- window ----

    fn read_buf(&mut self, input: &mut Input<'_>, at: usize, size: usize) -> usize {
        let taken = input.take(size);
        self.window[at..at + taken.len()].copy_from_slice(taken);
        self.adler = adler32(self.adler, taken);
        taken.len()
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
                self.window.copy_within(wsize..2 * wsize, 0);
                if self.match_start >= wsize {
                    self.match_start -= wsize;
                } else {
                    self.match_start = 0;
                    self.prev_length = 0;
                }
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
            let n = self.read_buf(input, at, more);
            self.lookahead += n;
            if self.lookahead + self.insert >= MIN_MATCH {
                let str = self.strstart - self.insert;
                if str >= 1 {
                    self.quick_insert_string(str + 2 - MIN_MATCH);
                }
                let mut count = self.insert;
                if self.lookahead == 1 {
                    count -= 1;
                }
                if count > 0 {
                    self.insert_string(str, count);
                    self.insert -= count;
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

    // ---- matching ----

    #[inline]
    fn compare256(&self, a: usize, b: usize) -> usize {
        let lhs = &self.window[a..a + 256];
        let rhs = &self.window[b..b + 256];
        lhs.iter().zip(rhs).take_while(|(x, y)| x == y).count()
    }

    /// `longest_match_c` (not the slow variant): sets `match_start`, returns the length.
    fn longest_match(&mut self, mut cur_match: usize) -> usize {
        let strstart = self.strstart;
        let wmask = self.w_mask;
        let lookahead = self.lookahead;
        let mut best_len = if self.prev_length != 0 {
            self.prev_length
        } else {
            MIN_MATCH - 1
        };
        let offset_for = |best: usize| {
            let mut offset = best - 1;
            if best >= 4 {
                offset -= 2;
                if best >= 8 {
                    offset -= 4;
                }
            }
            offset
        };
        let mut offset = offset_for(best_len);
        let mut chain_length = self.max_chain;
        if best_len >= self.good_match {
            chain_length >>= 2;
        }
        let nice_match = self.nice_match;
        let limit = if strstart > self.max_dist() {
            strstart - self.max_dist()
        } else {
            0
        };
        let early_exit = self.level < 5;
        loop {
            if cur_match >= strstart {
                break;
            }
            let width = if best_len < 4 {
                2
            } else if best_len >= 8 {
                8
            } else {
                4
            };
            loop {
                let end_ok = self.window[cur_match + offset..cur_match + offset + width]
                    == self.window[strstart + offset..strstart + offset + width];
                if end_ok
                    && self.window[cur_match..cur_match + width] == self.window[strstart..strstart + width]
                {
                    break;
                }
                chain_length -= 1;
                if chain_length != 0 {
                    cur_match = usize::from(self.prev[cur_match & wmask]);
                    if cur_match > limit {
                        continue;
                    }
                }
                return best_len;
            }
            let len = self.compare256(strstart + 2, cur_match + 2) + 2;
            if len > best_len {
                self.match_start = cur_match;
                if len >= lookahead {
                    return lookahead;
                }
                best_len = len;
                if best_len >= nice_match {
                    return best_len;
                }
                offset = offset_for(best_len);
            } else if early_exit {
                break;
            }
            chain_length -= 1;
            if chain_length != 0 {
                cur_match = usize::from(self.prev[cur_match & wmask]);
                if cur_match > limit {
                    continue;
                }
            }
            return best_len;
        }
        best_len
    }

    // ---- deflate_medium ----

    fn emit_match(&mut self, m: Candidate) -> bool {
        let mut bflush = false;
        if usize::from(m.match_length) < WANT_MIN_MATCH {
            let start = usize::from(m.strstart);
            for pos in start..start + usize::from(m.match_length) {
                bflush |= self.trees.tally_lit(self.window[pos]);
                self.lookahead -= 1;
            }
            return bflush;
        }
        bflush |= self.trees.tally_dist(
            usize::from(m.strstart) - usize::from(m.match_start),
            usize::from(m.match_length) - MIN_MATCH,
        );
        self.lookahead -= usize::from(m.match_length);
        bflush
    }

    fn insert_match(&mut self, mut m: Candidate) {
        if self.lookahead <= usize::from(m.match_length) + WANT_MIN_MATCH {
            return;
        }
        let org = usize::from(m.orgstart);
        if usize::from(m.match_length) < WANT_MIN_MATCH {
            m.strstart = m.strstart.wrapping_add(1);
            m.match_length = m.match_length.wrapping_sub(1);
            if m.match_length > 0 {
                let start = usize::from(m.strstart);
                let len = usize::from(m.match_length);
                if start >= org {
                    if start + len > org {
                        self.insert_string(start, len);
                    } else {
                        self.insert_string(start, org - start + 1);
                    }
                }
            }
            return;
        }
        if usize::from(m.match_length) <= 16 * self.max_lazy && self.lookahead >= WANT_MIN_MATCH {
            m.match_length -= 1;
            m.strstart = m.strstart.wrapping_add(1);
            let start = usize::from(m.strstart);
            let len = usize::from(m.match_length);
            if start >= org {
                if start + len > org {
                    self.insert_string(start, len);
                } else {
                    self.insert_string(start, org - start + 1);
                }
            } else if org < start + len {
                self.insert_string(org, start + len - org);
            }
        } else {
            let end = usize::from(m.strstart) + usize::from(m.match_length);
            if end >= MIN_MATCH - 2 {
                self.quick_insert_string(end + 2 - MIN_MATCH);
            }
        }
    }

    fn fizzle_matches(&self, current: &mut Candidate, next: &mut Candidate) {
        if current.match_length <= 1 {
            return;
        }
        let length = usize::from(current.match_length);
        if length > 1 + usize::from(next.match_start) || length > 1 + usize::from(next.strstart) {
            return;
        }
        let back = usize::from(current.match_length) - 1;
        if self.window[usize::from(next.match_start) - back] != self.window[usize::from(next.strstart) - back]
        {
            return;
        }
        let mut c = *current;
        let mut n = *next;
        let max_dist = self.max_dist();
        let limit = if usize::from(next.strstart) > max_dist {
            usize::from(next.strstart) - max_dist
        } else {
            0
        };
        let mut changed = 0;
        let mut mi = usize::from(n.match_start) - 1;
        let mut oi = usize::from(n.strstart) - 1;
        while self.window[mi] == self.window[oi] {
            if c.match_length < 1
                || usize::from(n.strstart) <= limit
                || n.match_length >= 256
                || n.match_start <= 1
            {
                break;
            }
            n.strstart -= 1;
            n.match_start -= 1;
            n.match_length += 1;
            c.match_length -= 1;
            mi -= 1;
            oi -= 1;
            changed += 1;
        }
        if changed == 0 {
            return;
        }
        if c.match_length <= 1 && n.match_length != 2 {
            n.orgstart += 1;
            *current = c;
            *next = n;
        }
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

    fn deflate_medium(&mut self, input: &mut Input<'_>, out: &mut Output, flush: Flush) -> BlockState {
        let mut current = Candidate::default();
        let mut next = Candidate::default();
        let early_exit = self.level < 5;
        loop {
            if self.lookahead < MIN_LOOKAHEAD {
                self.fill_window(input);
                if self.lookahead < MIN_LOOKAHEAD && flush == Flush::None {
                    return BlockState::NeedMore;
                }
                if self.lookahead == 0 {
                    break;
                }
                next.match_length = 0;
            }
            if !early_exit && next.match_length > 0 {
                current = next;
                next.match_length = 0;
            } else {
                let mut hash_head = 0;
                if self.lookahead >= WANT_MIN_MATCH {
                    hash_head = self.quick_insert_string(self.strstart);
                }
                current.strstart = self.strstart as u16;
                current.orgstart = current.strstart;
                let dist = self.strstart as i64 - hash_head as i64;
                if dist <= self.max_dist() as i64 && dist > 0 && hash_head != 0 {
                    current.match_length = self.longest_match(hash_head) as u16;
                    current.match_start = self.match_start as u16;
                    if usize::from(current.match_length) < WANT_MIN_MATCH {
                        current.match_length = 1;
                    }
                    if current.match_start >= current.strstart {
                        current.match_length = 1;
                    }
                } else {
                    current.match_start = 0;
                    current.match_length = 1;
                }
            }
            self.insert_match(current);
            let ahead = usize::from(current.strstart) + usize::from(current.match_length);
            if !early_exit && self.lookahead > MIN_LOOKAHEAD && ahead < self.window_size - MIN_LOOKAHEAD {
                self.strstart = ahead;
                let hash_head = self.quick_insert_string(self.strstart);
                next.strstart = self.strstart as u16;
                next.orgstart = next.strstart;
                let dist = self.strstart as i64 - hash_head as i64;
                if dist <= self.max_dist() as i64 && dist > 0 && hash_head != 0 {
                    next.match_length = self.longest_match(hash_head) as u16;
                    next.match_start = self.match_start as u16;
                    if next.match_start >= next.strstart {
                        next.match_length = 1;
                    }
                    if usize::from(next.match_length) < WANT_MIN_MATCH {
                        next.match_length = 1;
                    } else {
                        self.fizzle_matches(&mut current, &mut next);
                    }
                } else {
                    next.match_start = 0;
                    next.match_length = 1;
                }
                self.strstart = usize::from(current.strstart);
            } else {
                next.match_length = 0;
            }
            let bflush = self.emit_match(current);
            self.strstart += usize::from(current.match_length);
            if bflush {
                self.flush_block_only(false, out);
                if out.avail == 0 {
                    return BlockState::NeedMore;
                }
            }
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

    // ---- stream ----

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
                0 | 1 => 0,
                2..=5 => 1,
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
            let bstate = self.deflate_medium(input, out, flush);
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
