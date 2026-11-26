//! Runtime instruction-set dispatch for autovectorized loops (AVX2/AVX-512 when available).
//!
//! The work must be a function item marked `#[inline(always)]` so that it is compiled inside
//! pulp's target-feature wrapper; closures would be emitted as separate baseline functions.
//! Floating-point results are unchanged: Rust never contracts `a * b + c` into an FMA.

struct Call<F, A>(F, A);

impl<R, A, F: FnOnce(A) -> R> pulp::WithSimd for Call<F, A> {
    type Output = R;

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, _simd: S) -> R {
        (self.0)(self.1)
    }
}

/// Run `f(args)` with the widest detected SIMD target features enabled (small bodies only:
/// the call goes through `FnOnce::call_once`, which LLVM may decline to inline).
#[inline(always)]
pub(crate) fn vectorized<A, R>(f: impl FnOnce(A) -> R, args: A) -> R {
    pulp::Arch::new().dispatch(Call(f, args))
}

/// A unit of work whose `#[inline(always)]` body is compiled inside the dispatch wrapper.
pub(crate) trait Kernel {
    type Output;
    fn run(self) -> Self::Output;
}

struct Wrap<K>(K);

impl<K: Kernel> pulp::WithSimd for Wrap<K> {
    type Output = K::Output;

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, _simd: S) -> K::Output {
        self.0.run()
    }
}

/// Run a [`Kernel`] with the widest detected SIMD target features enabled.
#[inline(always)]
pub(crate) fn dispatch<K: Kernel>(kernel: K) -> K::Output {
    pulp::Arch::new().dispatch(Wrap(kernel))
}
