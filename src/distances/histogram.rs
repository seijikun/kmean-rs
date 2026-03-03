use crate::memory::SupportedSimdArray;
use crate::{DistanceFunction, Primitive};
use std::simd::Simd;

pub struct HistogramDistance;

impl<T, const LANES: usize> DistanceFunction<T, LANES> for HistogramDistance
where
    T: Primitive,
    Simd<T, LANES>: SupportedSimdArray<T, LANES>,
{
    #[inline(always)]
    fn distance(&self, a: &[T], b: &[T]) -> T {
        let mut total = T::zero();
        let mut cdf_a = T::zero();
        let mut cdf_b = T::zero();
        for (x, y) in a.iter().zip(b.iter()) {
            cdf_a += x;
            cdf_b += y;
            total += (cdf_a - cdf_b).abs();
        }
        total
    }
}
