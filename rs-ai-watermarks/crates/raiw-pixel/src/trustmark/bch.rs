//! TrustMark's BCH codec (`trustmark/bchecc.py`), transcribed with its quirks.
//!
//! The Python port of the Linux kernel BCH library differs from the kernel in ways that change
//! verdicts (a degree-2 locator without roots reports zero flips, list indices can wrap), so this
//! keeps its control flow and index arithmetic. Out-of-range list accesses, which raise
//! `IndexError` in Python, surface as [`BchError`].

/// A list index Python would have rejected with `IndexError`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BchError;

/// Python list indexing: negative indices count from the end.
fn at(list: &[i64], index: i64) -> Result<i64, BchError> {
    let len = list.len() as i64;
    let i = if index < 0 { index + len } else { index };
    if (0..len).contains(&i) { Ok(list[i as usize]) } else { Err(BchError) }
}

fn slot(list: &mut [i64], index: i64) -> Result<&mut i64, BchError> {
    let len = list.len() as i64;
    let i = if index < 0 { index + len } else { index };
    if (0..len).contains(&i) { Ok(&mut list[i as usize]) } else { Err(BchError) }
}

fn deg(mut x: i64) -> i64 {
    let mut count = 0;
    while x >> 1 != 0 {
        x >>= 1;
        count += 1;
    }
    count
}

const fn ceilop(a: i64, b: i64) -> i64 {
    (a + b - 1) / b
}

fn load4(bytes: &[u8]) -> i64 {
    (i64::from(bytes[0]) << 24) | (i64::from(bytes[1]) << 16) | (i64::from(bytes[2]) << 8) | i64::from(bytes[3])
}

#[derive(Clone, Debug)]
struct Polynomial {
    deg: i64,
    c: Vec<i64>,
}

/// One BCH code over GF(2^m) correcting `t` bit errors.
#[derive(Clone, Debug)]
pub struct Bch {
    m: i64,
    t: i64,
    n: i64,
    ecc_bits: i64,
    ecc_bytes: i64,
    exponents: Vec<i64>,
    logarithms: Vec<i64>,
    elp_pre: Vec<i64>,
    cyclic_tab: Vec<i64>,
}

impl Bch {
    /// `BCH(t, poly)`; `None` when `poly` is not primitive (Python returns early).
    pub fn new(t: i64, poly: i64) -> Option<Self> {
        let m = deg(poly);
        let n = (1 << m) - 1;
        let words = ceilop(m * t, 32);
        let mut bch = Self {
            m,
            t,
            n,
            ecc_bits: 0,
            ecc_bytes: ceilop(m * t, 8),
            exponents: vec![0; (n + 1) as usize],
            logarithms: vec![0; (n + 1) as usize],
            elp_pre: vec![0; (m + 1) as usize],
            cyclic_tab: vec![0; (words * 1024) as usize],
        };
        let k = 1 << deg(poly);
        let mut x = 1;
        for i in 0..n {
            bch.exponents[i as usize] = x;
            bch.logarithms[x as usize] = i;
            if i != 0 && x == 1 {
                return None;
            }
            x *= 2;
            if x & k != 0 {
                x ^= poly;
            }
        }
        bch.logarithms[0] = 0;
        bch.exponents[n as usize] = 1;

        // Generator polynomial from the cyclotomic cosets of 1, 3, ..., 2t - 1.
        let mut roots = vec![false; (n + 1) as usize];
        for i in 0..t {
            let mut r = 2 * i + 1;
            for _ in 0..m {
                roots[r as usize] = true;
                r = bch.modulo(2 * r);
            }
        }
        let mut g = Polynomial { deg: 0, c: vec![0; (m * t + 1) as usize] };
        g.c[0] = 1;
        for i in 0..n {
            if roots[i as usize] {
                let r = bch.exponents[i as usize];
                g.c[(g.deg + 1) as usize] = 1;
                for j in (1..=g.deg).rev() {
                    g.c[j as usize] = bch.g_mul(g.c[j as usize], r) ^ g.c[(j - 1) as usize];
                }
                g.c[0] = bch.g_mul(g.c[0], r);
                g.deg += 1;
            }
        }
        let mut genpoly = vec![0_i64; ceilop(m * t + 1, 32) as usize];
        let (mut remaining, mut word_index) = (g.deg + 1, 0);
        while remaining > 0 {
            let nbits = remaining.min(32);
            let mut word = 0;
            for j in 0..nbits {
                if g.c[(remaining - 1 - j) as usize] != 0 {
                    word |= 1 << (31 - j);
                }
            }
            genpoly[word_index] = word;
            word_index += 1;
            remaining -= nbits;
        }
        bch.ecc_bits = g.deg;
        bch.build_cyclic(&genpoly);

        // Precomputed solutions of x^2 + x = 2^i for the degree-2 root finder.
        let (mut sum, mut aexp) = (0, 0);
        'outer: for i in 0..m {
            for j in 0..m {
                sum ^= bch.g_pow(i * (1 << j));
            }
            if sum != 0 {
                aexp = bch.exponents[i as usize];
                break 'outer;
            }
        }
        let (mut x, mut remaining, mut precomp) = (0, m, [false; 31]);
        while x <= n && remaining != 0 {
            let mut y = bch.g_sqrt(x) ^ x;
            for _ in 0..2 {
                let r = bch.g_log(y);
                if y != 0 && r < m && !precomp[r as usize] {
                    bch.elp_pre[r as usize] = x;
                    precomp[r as usize] = true;
                    remaining -= 1;
                    break;
                }
                y ^= aexp;
            }
            x += 1;
        }
        Some(bch)
    }

    pub const fn t(&self) -> i64 {
        self.t
    }

    pub const fn ecc_bits(&self) -> i64 {
        self.ecc_bits
    }

    pub const fn ecc_bytes(&self) -> i64 {
        ceilop(self.m * self.t, 8)
    }

    fn modulo(&self, v: i64) -> i64 {
        if v < self.n { v } else { v - self.n }
    }

    fn modn(&self, mut v: i64) -> i64 {
        while v >= self.n {
            v -= self.n;
            v = (v & self.n) + (v >> self.m);
        }
        v
    }

    fn g_log(&self, x: i64) -> i64 {
        self.logarithms[x as usize]
    }

    fn g_pow(&self, i: i64) -> i64 {
        at(&self.exponents, self.modn(i)).unwrap_or(0)
    }

    fn g_mul(&self, a: i64, b: i64) -> i64 {
        if a > 0 && b > 0 {
            self.exponents[self.modulo(self.logarithms[a as usize] + self.logarithms[b as usize]) as usize]
        } else {
            0
        }
    }

    fn g_div(&self, a: i64, b: i64) -> i64 {
        if a == 0 {
            return 0;
        }
        let index = self.modulo(self.logarithms[a as usize] + self.n - self.logarithms[b as usize]);
        self.exponents[index as usize]
    }

    fn g_sqrt(&self, a: i64) -> i64 {
        if a == 0 { 0 } else { self.exponents[self.modulo(2 * self.logarithms[a as usize]) as usize] }
    }

    fn build_cyclic(&mut self, g: &[i64]) {
        let l = ceilop(self.m * self.t, 32);
        let plen = ceilop(self.ecc_bits + 1, 32);
        let ecclen = ceilop(self.ecc_bits, 32);
        self.cyclic_tab = vec![0; (4 * 256 * l) as usize];
        for i in 0..256 {
            for b in 0..4 {
                let offset = (b * 256 + i) * l;
                let mut data: i64 = i << (8 * b);
                while data != 0 {
                    let d = deg(data);
                    data ^= g[0] >> (31 - d);
                    for j in 0..ecclen {
                        let hi = if d < 31 { (g[j as usize] << (d + 1)) & 0xFFFF_FFFF } else { 0 };
                        let lo = if j + 1 < plen { g[(j + 1) as usize] >> (31 - d) } else { 0 };
                        self.cyclic_tab[(j + offset) as usize] ^= hi | lo;
                    }
                }
            }
        }
    }

    /// ECC bytes for `data`, plus the 62-word remainder buffer Python keeps as `ecc_buf`.
    fn encode_words(&self, data: &[u8]) -> Vec<i64> {
        let l = ceilop(self.m * self.t, 32) - 1;
        let stride = l + 1;
        let tab = |index: i64| self.cyclic_tab[index as usize];
        let mut r = vec![0_i64; ceilop(31 * 64, 32) as usize];
        let whole = data.len() / 4 * 4;
        for chunk in data[..whole].chunks_exact(4) {
            let w = load4(chunk) ^ r[0];
            let p0 = stride * (w & 0xFF);
            let p1 = 256 * stride + stride * ((w >> 8) & 0xFF);
            let p2 = 2 * 256 * stride + stride * ((w >> 16) & 0xFF);
            let p3 = 3 * 256 * stride + stride * ((w >> 24) & 0xFF);
            for i in 0..l {
                r[i as usize] = r[(i + 1) as usize] ^ tab(p0 + i) ^ tab(p1 + i) ^ tab(p2 + i) ^ tab(p3 + i);
            }
            r[l as usize] = tab(p0 + l) ^ tab(p1 + l) ^ tab(p2 + l) ^ tab(p3 + l);
        }
        for &byte in &data[whole..] {
            let mut pidx = stride * (((r[0] >> 24) ^ i64::from(byte)) & 0xFF);
            for i in 0..l {
                let (cur, next) = (r[i as usize], r[(i + 1) as usize]);
                r[i as usize] = (((cur << 8) & 0xFFFF_FFFF) | (next >> 24)) ^ tab(pidx);
                pidx += 1;
            }
            r[l as usize] = ((r[l as usize] << 8) & 0xFFFF_FFFF) ^ tab(pidx);
        }
        r
    }

    /// The ECC bytes `encode` returns.
    pub fn encode(&self, data: &[u8]) -> Vec<u8> {
        let words = self.encode_words(data);
        words.iter().flat_map(|w| (*w as u32).to_be_bytes()).take(self.ecc_bytes as usize).collect()
    }

    /// Correct `data` in place against `recv_ecc`; the number of flips, or -1 when uncorrectable.
    pub fn decode(&self, data: &mut [u8], recv_ecc: &[u8]) -> Result<i64, BchError> {
        let mut ecc_buf = self.encode_words(data);
        let whole = recv_ecc.len() / 4 * 4;
        let mut eccbuf: Vec<i64> = recv_ecc[..whole].chunks_exact(4).map(load4).collect();
        let tail = &recv_ecc[whole..];
        let recv_len = if tail.is_empty() {
            0
        } else {
            let mut padded = [0_u8; 4];
            padded[..tail.len()].copy_from_slice(tail);
            eccbuf.push(load4(&padded));
            4
        };
        let eccwords = ceilop(self.m * self.t, 32);
        let mut sum = 0;
        for i in 0..eccwords {
            let word = at(&eccbuf, i)?;
            let entry = slot(&mut ecc_buf, i)?;
            *entry ^= word;
            sum |= *entry;
        }
        if sum == 0 {
            return Ok(0);
        }

        // Syndromes.
        let t = self.t;
        let mut s = self.ecc_bits;
        let mut syn = vec![0_i64; (2 * t) as usize];
        let partial = s & 31;
        if partial != 0 {
            *slot(&mut ecc_buf, s / 32)? &= !((1_i64 << (32 - partial)) - 1);
        }
        let mut synptr = 0;
        while s > 0 || synptr == 0 {
            let mut poly = at(&ecc_buf, synptr)?;
            synptr += 1;
            s -= 32;
            while poly != 0 {
                let i = deg(poly);
                for j in (0..2 * t).step_by(2) {
                    syn[j as usize] ^= self.g_pow((j + 1) * (i + s));
                }
                poly ^= 1 << i;
            }
        }
        for i in 0..t {
            syn[(2 * i + 1) as usize] = self.g_sqrt(syn[i as usize]);
        }

        // Berlekamp-Massey.
        let n = self.n;
        let (mut pp, mut pd) = (-1_i64, 1_i64);
        let mut pelp = Polynomial { deg: 0, c: vec![0; (2 * t) as usize] };
        pelp.c[0] = 1;
        let mut elp = pelp.clone();
        let mut d = syn[0];
        for i in 0..t {
            if elp.deg > t {
                break;
            }
            if d != 0 {
                let k = 2 * i - pp;
                let elp_copy = elp.clone();
                let tmp = self.g_log(d) + n - self.g_log(pd);
                for j in 0..=pelp.deg {
                    let coefficient = at(&pelp.c, j)?;
                    if coefficient != 0 {
                        let l = self.g_log(coefficient);
                        *slot(&mut elp.c, j + k)? ^= self.g_pow(tmp + l);
                    }
                }
                let tmp = pelp.deg + k;
                if tmp > elp.deg {
                    elp.deg = tmp;
                    pelp = elp_copy;
                    pd = d;
                    pp = 2 * i;
                }
            }
            if i < t - 1 {
                d = at(&syn, 2 * i + 2)?;
                for j in 1..=elp.deg {
                    d ^= self.g_mul(at(&elp.c, j)?, at(&syn, 2 * i + 2 - j)?);
                }
            }
        }

        let errloc = self.roots(data.len() as i64, &elp)?;
        let Some(mut errloc) = errloc else {
            return Ok(-1);
        };
        let nbits = data.len() as i64 * 8 + self.ecc_bits;
        for loc in &mut errloc {
            if *loc >= nbits {
                return Ok(-1);
            }
            *loc = nbits - 1 - *loc;
            *loc = (*loc & !7) | (7 - (*loc & 7));
        }
        let limit = (data.len() as i64 + recv_len) * 8;
        for &bitflip in &errloc {
            let (byte, bit) = (bitflip / 8, 1_u8 << (bitflip & 7));
            if bitflip < limit && byte < data.len() as i64 {
                data[byte as usize] ^= bit;
            }
        }
        Ok(errloc.len() as i64)
    }

    /// Error locations (`getroots`); `None` when the locator has too few roots.
    fn roots(&self, k: i64, poly: &Polynomial) -> Result<Option<Vec<i64>>, BchError> {
        let mut roots = Vec::new();
        let n = self.n;
        if poly.deg > 2 {
            let k = k * 8 + self.ecc_bits;
            let mut rep = vec![0_i64; (2 * self.t) as usize];
            let top = at(&poly.c, poly.deg)?;
            let l = n - self.g_log(top);
            for i in 0..poly.deg {
                let c = at(&poly.c, i)?;
                *slot(&mut rep, i)? = if c != 0 { self.modulo(self.g_log(c) + l) } else { -1 };
            }
            *slot(&mut rep, poly.deg)? = 0;
            let syn0 = self.g_div(at(&poly.c, 0)?, top);
            for i in (n - k + 1)..=n {
                let mut syn = syn0;
                for j in 1..=poly.deg {
                    let m = at(&rep, j)?;
                    if m >= 0 {
                        syn ^= self.g_pow(m + j * i);
                    }
                }
                if syn == 0 {
                    roots.push(n - i);
                    if roots.len() as i64 == poly.deg {
                        break;
                    }
                }
            }
            if (roots.len() as i64) < poly.deg {
                return Ok(None);
            }
        }
        if poly.deg == 1 {
            let (c0, c1) = (at(&poly.c, 0)?, at(&poly.c, 1)?);
            if c0 != 0 {
                roots.push(self.modulo(n - self.logarithms[c0 as usize] + self.logarithms[c1 as usize]));
            }
        }
        if poly.deg == 2 {
            let (c0, c1, c2) = (at(&poly.c, 0)?, at(&poly.c, 1)?, at(&poly.c, 2)?);
            if c0 != 0 && c1 != 0 {
                let (l0, l1, l2) = (self.g_log(c0), self.g_log(c1), self.g_log(c2));
                let u = self.g_pow(l0 + l2 + 2 * (n - l1));
                let (mut r, mut v) = (0, u);
                while v != 0 {
                    let i = deg(v);
                    r ^= at(&self.elp_pre, i)?;
                    v ^= 1 << i;
                }
                if self.g_sqrt(r) ^ r == u {
                    roots.push(self.modn(2 * n - l1 - self.g_log(r) + l2));
                    roots.push(self.modn(2 * n - l1 - self.g_log(r ^ 1) + l2));
                }
            }
        }
        Ok(Some(roots))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_have_python_sizes() {
        for (t, bits, bytes) in [(5, 35, 5), (4, 28, 4), (3, 21, 3), (8, 56, 7)] {
            let bch = Bch::new(t, 137).expect("primitive");
            assert_eq!((bch.ecc_bits(), bch.ecc_bytes()), (bits, bytes), "t={t}");
        }
    }

    #[test]
    fn corrects_up_to_t_flips() {
        let bch = Bch::new(5, 137).expect("primitive");
        let data = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];
        let ecc = bch.encode(&data);
        let mut corrupted = data;
        for bit in [3, 17, 30, 41, 62] {
            corrupted[bit / 8] ^= 0x80 >> (bit % 8);
        }
        assert_eq!(bch.decode(&mut corrupted, &ecc), Ok(5));
        assert_eq!(corrupted, data);
    }
}
