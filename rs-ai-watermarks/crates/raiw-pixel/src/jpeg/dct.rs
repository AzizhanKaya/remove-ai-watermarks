//! libjpeg-turbo's accurate integer DCTs (`jfdctint.c`, `jidctint.c`) and quantizer (`jcdctmgr.c`).

const CONST_BITS: u32 = 13;
const PASS1_BITS: u32 = 2;
const FIX_0_298631336: i64 = 2446;
const FIX_0_390180644: i64 = 3196;
const FIX_0_541196100: i64 = 4433;
const FIX_0_765366865: i64 = 6270;
const FIX_0_899976223: i64 = 7373;
const FIX_1_175875602: i64 = 9633;
const FIX_1_501321110: i64 = 12299;
const FIX_1_847759065: i64 = 15137;
const FIX_1_961570560: i64 = 16069;
const FIX_2_053119869: i64 = 16819;
const FIX_2_562915447: i64 = 20995;
const FIX_3_072711026: i64 = 25172;

#[inline]
const fn descale(x: i64, n: u32) -> i64 {
    (x + (1 << (n - 1))) >> n
}

/// The LL&M odd part shared by both transforms: `(o7, o5, o3, o1)` from `(t4, t5, t6, t7)`.
#[inline]
fn odd(t4: i64, t5: i64, t6: i64, t7: i64) -> (i64, i64, i64, i64) {
    let (z1, z2, z3, z4) = (t4 + t7, t5 + t6, t4 + t6, t5 + t7);
    let z5 = (z3 + z4) * FIX_1_175875602;
    let (t4, t5, t6, t7) = (t4 * FIX_0_298631336, t5 * FIX_2_053119869, t6 * FIX_3_072711026, t7 * FIX_1_501321110);
    let (z1, z2) = (z1 * -FIX_0_899976223, z2 * -FIX_2_562915447);
    let (z3, z4) = (z3 * -FIX_1_961570560 + z5, z4 * -FIX_0_390180644 + z5);
    (t4 + z1 + z3, t5 + z2 + z4, t6 + z2 + z3, t7 + z1 + z4)
}

/// `jpeg_fdct_islow` on level-shifted samples; output scaled up by 8.
pub fn forward(block: &mut [i64; 64]) {
    for row in 0..8 {
        let d = &mut block[row * 8..row * 8 + 8];
        let (t0, t7) = (d[0] + d[7], d[0] - d[7]);
        let (t1, t6) = (d[1] + d[6], d[1] - d[6]);
        let (t2, t5) = (d[2] + d[5], d[2] - d[5]);
        let (t3, t4) = (d[3] + d[4], d[3] - d[4]);
        let (t10, t13, t11, t12) = (t0 + t3, t0 - t3, t1 + t2, t1 - t2);
        d[0] = (t10 + t11) << PASS1_BITS;
        d[4] = (t10 - t11) << PASS1_BITS;
        let z1 = (t12 + t13) * FIX_0_541196100;
        d[2] = descale(z1 + t13 * FIX_0_765366865, CONST_BITS - PASS1_BITS);
        d[6] = descale(z1 + t12 * -FIX_1_847759065, CONST_BITS - PASS1_BITS);
        let (o7, o5, o3, o1) = odd(t4, t5, t6, t7);
        d[7] = descale(o7, CONST_BITS - PASS1_BITS);
        d[5] = descale(o5, CONST_BITS - PASS1_BITS);
        d[3] = descale(o3, CONST_BITS - PASS1_BITS);
        d[1] = descale(o1, CONST_BITS - PASS1_BITS);
    }
    for col in 0..8 {
        let v = |k: usize| block[k * 8 + col];
        let (t0, t7) = (v(0) + v(7), v(0) - v(7));
        let (t1, t6) = (v(1) + v(6), v(1) - v(6));
        let (t2, t5) = (v(2) + v(5), v(2) - v(5));
        let (t3, t4) = (v(3) + v(4), v(3) - v(4));
        let (t10, t13, t11, t12) = (t0 + t3, t0 - t3, t1 + t2, t1 - t2);
        let z1 = (t12 + t13) * FIX_0_541196100;
        let (o7, o5, o3, o1) = odd(t4, t5, t6, t7);
        let n = CONST_BITS + PASS1_BITS;
        let out = [
            descale(t10 + t11, PASS1_BITS),
            descale(o1, n),
            descale(z1 + t13 * FIX_0_765366865, n),
            descale(o3, n),
            descale(t10 - t11, PASS1_BITS),
            descale(o5, n),
            descale(z1 + t12 * -FIX_1_847759065, n),
            descale(o7, n),
        ];
        for (k, value) in out.into_iter().enumerate() {
            block[k * 8 + col] = value;
        }
    }
}

/// Reciprocal quantizer entry for one divisor (`compute_reciprocal`, 16-bit `DCTELEM`).
#[derive(Clone, Copy, Debug)]
pub struct Divisor {
    recip: u32,
    corr: u32,
    shift: u32,
}

impl Divisor {
    pub fn new(divisor: u32) -> Self {
        if divisor == 1 {
            return Self { recip: 1, corr: 0, shift: 0 };
        }
        let b = 31 - divisor.leading_zeros();
        let mut r = 16 + b;
        let mut fq = (1_u64 << r) / u64::from(divisor);
        let fr = (1_u64 << r) % u64::from(divisor);
        let mut c = divisor / 2;
        if fr == 0 {
            fq >>= 1;
            r -= 1;
        } else if fr <= u64::from(divisor / 2) {
            c += 1;
        } else {
            fq += 1;
        }
        Self { recip: fq as u32, corr: c, shift: r - 16 }
    }

    #[inline]
    pub fn quantize(self, value: i64) -> i32 {
        let magnitude = value.unsigned_abs() as u32;
        let product = magnitude.wrapping_add(self.corr).wrapping_mul(self.recip) >> (self.shift + 16);
        let q = i32::from(product as u16 as i16);
        if value < 0 { -q } else { q }
    }
}

/// `jpeg_idct_islow`: dequantize, invert, and range-limit through libjpeg's wrapping table.
pub fn inverse(coefficients: &[i32; 64], quant: &[u16; 64], out: &mut [u8; 64]) {
    let dq = |k: usize| i64::from(coefficients[k]) * i64::from(quant[k]);
    let mut ws = [0_i64; 64];
    for col in 0..8 {
        if (1..8).all(|k| coefficients[k * 8 + col] == 0) {
            let dc = dq(col) << PASS1_BITS;
            (0..8).for_each(|k| ws[k * 8 + col] = dc);
            continue;
        }
        let rows = even_odd([dq(col), dq(8 + col), dq(16 + col), dq(24 + col), dq(32 + col), dq(40 + col), dq(48 + col), dq(56 + col)]);
        for (k, value) in rows.into_iter().enumerate() {
            ws[k * 8 + col] = descale(value, CONST_BITS - PASS1_BITS) as i32 as i64;
        }
    }
    for row in 0..8 {
        let w = &ws[row * 8..row * 8 + 8];
        let o = &mut out[row * 8..row * 8 + 8];
        if w[1..].iter().all(|&v| v == 0) {
            o.fill(range_limit(descale(w[0], PASS1_BITS + 3)));
            continue;
        }
        let values = even_odd([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]]);
        for (slot, value) in o.iter_mut().zip(values) {
            *slot = range_limit(descale(value, CONST_BITS + PASS1_BITS + 3));
        }
    }
}

/// The shared 1-D inverse butterfly, outputs in natural order before descaling.
#[inline]
fn even_odd(x: [i64; 8]) -> [i64; 8] {
    let (z2, z3) = (x[2], x[6]);
    let z1 = (z2 + z3) * FIX_0_541196100;
    let tmp2 = z1 + z3 * -FIX_1_847759065;
    let tmp3 = z1 + z2 * FIX_0_765366865;
    let tmp0 = (x[0] + x[4]) << CONST_BITS;
    let tmp1 = (x[0] - x[4]) << CONST_BITS;
    let (t10, t13, t11, t12) = (tmp0 + tmp3, tmp0 - tmp3, tmp1 + tmp2, tmp1 - tmp2);
    let (o0, o1, o2, o3) = odd(x[7], x[5], x[3], x[1]);
    [t10 + o3, t11 + o2, t12 + o1, t13 + o0, t13 - o0, t12 - o1, t11 - o2, t10 - o3]
}

/// libjpeg's post-IDCT `range_limit[x & RANGE_MASK]` for a centered sample.
#[inline]
fn range_limit(x: i64) -> u8 {
    match x & 1023 {
        v @ 0..=127 => (v + 128) as u8,
        128..=511 => 255,
        512..=895 => 0,
        v => (v - 896) as u8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reciprocal_matches_rounded_division() {
        for divisor in [8_u32, 16, 24, 80, 88, 792, 2040] {
            let d = Divisor::new(divisor);
            for value in -9000_i64..9000 {
                let expected = (value.abs() + i64::from(divisor / 2)) / i64::from(divisor);
                assert_eq!(i64::from(d.quantize(value)).abs(), expected, "{value}/{divisor}");
            }
        }
    }

    #[test]
    fn flat_block_round_trips() {
        let mut block = [37_i64; 64];
        forward(&mut block);
        assert_eq!(block[0], 37 * 64);
        assert!(block[1..].iter().all(|&v| v == 0));
        let mut coefficients = [0_i32; 64];
        coefficients[0] = (37 * 64 + 4) / 8;
        let mut out = [0_u8; 64];
        inverse(&coefficients, &[1; 64], &mut out);
        assert!(out.iter().all(|&v| v == 165));
    }
}
