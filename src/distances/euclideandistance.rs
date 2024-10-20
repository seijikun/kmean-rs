use crate::memory::SupportedSimdArray;
use crate::{DistanceFunction, Primitive};
use std::simd::num::SimdFloat;
use std::simd::{LaneCount, Simd, SupportedLaneCount};

pub struct EuclideanDistance;

impl<T, const LANES: usize> DistanceFunction<T, LANES> for EuclideanDistance
where
    T: Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: SupportedSimdArray<T, LANES>,
{
    #[inline(always)]
    fn distance(&self, a: &[T], b: &[T]) -> T {
        a.chunks_exact(LANES)
            .map(|i| Simd::from_slice(i))
            .zip(b.chunks_exact(LANES).map(|i| Simd::from_slice(i)))
            .map(|(sp, cp)| sp - cp)
            .map(|v| v * v)
            .sum::<Simd<T, LANES>>()
            .reduce_sum()
    }
}
