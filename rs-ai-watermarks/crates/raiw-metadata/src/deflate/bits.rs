//! LSB-first bit writer over the pending output buffer.

/// Pending output plus the partial byte not yet written.
#[derive(Debug, Default)]
pub struct BitWriter {
    pub pending: Vec<u8>,
    buf: u64,
    valid: u32,
}

impl BitWriter {
    #[inline]
    pub fn send_bits(&mut self, value: u64, len: u32) {
        if len > 32 {
            self.send_bits(value & 0xffff_ffff, 32);
            self.send_bits(value >> 32, len - 32);
            return;
        }
        self.buf |= value << self.valid;
        self.valid += len;
        if self.valid >= 32 {
            self.pending.extend_from_slice(&(self.buf as u32).to_le_bytes());
            self.buf >>= 32;
            self.valid -= 32;
        }
    }

    /// Write every complete byte (`zng_tr_flush_bits`); at most 7 bits stay buffered.
    pub fn flush_bits(&mut self) {
        while self.valid >= 8 {
            self.pending.push(self.buf as u8);
            self.buf >>= 8;
            self.valid -= 8;
        }
    }

    /// Pad to a byte boundary (`bi_windup`).
    pub fn windup(&mut self) {
        self.flush_bits();
        if self.valid > 0 {
            self.pending.push(self.buf as u8);
        }
        self.buf = 0;
        self.valid = 0;
    }

    pub fn put_u16_le(&mut self, value: u16) {
        self.pending.extend_from_slice(&value.to_le_bytes());
    }

    pub fn put_u16_be(&mut self, value: u16) {
        self.pending.extend_from_slice(&value.to_be_bytes());
    }

    pub fn put_u32_be(&mut self, value: u32) {
        self.pending.extend_from_slice(&value.to_be_bytes());
    }

    pub fn put_bytes(&mut self, data: &[u8]) {
        self.pending.extend_from_slice(data);
    }
}

/// Adler-32 of `data` continued from `adler`.
pub fn adler32(adler: u32, data: &[u8]) -> u32 {
    const BASE: u32 = 65521;
    const NMAX: usize = 5552;
    let mut a = adler & 0xffff;
    let mut b = adler >> 16;
    for chunk in data.chunks(NMAX) {
        for &byte in chunk {
            a += u32::from(byte);
            b += a;
        }
        a %= BASE;
        b %= BASE;
    }
    (b << 16) | a
}
