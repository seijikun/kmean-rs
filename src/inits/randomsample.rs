use crate::{KMeans, KMeansState, KMeansConfig, memory::*};
use rand::prelude::*;
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Sub};
use std::simd::{num::SimdFloat, LaneCount, Simd, SimdElement, SupportedLaneCount};
use std::ops::DerefMut;

#[inline(always)] pub fn calculate<'a, T, const LANES: usize>(kmean: &KMeans<T, LANES>, state: &mut KMeansState<T>, config: &KMeansConfig<'a, T>)
				where
    T: SimdElement
        + Copy
        + Default
        + Add<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Sub<Output = T>
        + Sum
        + Primitive,
    Simd<T, LANES>: Sub<Output = Simd<T, LANES>>
        + Add<Output = Simd<T, LANES>>
        + Sub<Output = Simd<T, LANES>>
        + Mul<Output = Simd<T, LANES>>
        + Div<Output = Simd<T, LANES>>
        + Sum
        + SimdFloat<Scalar = T>,
    LaneCount<LANES>: SupportedLaneCount,
{
    kmean.p_samples.chunks_exact(kmean.p_sample_dims)
		.choose_multiple(config.rnd.borrow_mut().deref_mut(), state.k).iter().cloned()
		.enumerate()
		.for_each(|(ci, c)| { // Copy randomly chosen centroids into state.centroids
			state.set_centroid_from_iter(ci, c.iter().cloned());
		});
}