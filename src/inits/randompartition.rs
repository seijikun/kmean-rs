use crate::{KMeans, KMeansState, KMeansConfig, memory::*};
use rand::prelude::*;
use std::simd::{Simd, SimdElement, LaneCount, SupportedLaneCount, num::SimdFloat};
use std::iter::Sum;
use std::ops::{Add, Mul, Div, Sub};

// TODO
#[inline(always)] pub fn calculate<'a, T, const LANES: usize>(
    kmean: &KMeans<T, LANES>,
    state: &mut KMeansState<T>,
    config: &KMeansConfig<'a, T>,
) where
    T: SimdElement + Copy + Default + Add<Output = T> + Mul<Output = T> + Div<Output = T> + Sub<Output = T> + Sum + Primitive,
    Simd<T, LANES>: Sub<Output = Simd<T, LANES>>
        + Add<Output = Simd<T, LANES>>
        + Mul<Output = Simd<T, LANES>>
        + Div<Output = Simd<T, LANES>>
        + Sum
        + SimdFloat<Scalar = T>,
    LaneCount<LANES>: SupportedLaneCount,
{
	let (assignments, centroids, centroid_frequency, k) =
		(&mut state.assignments, &mut state.centroids, &mut state.centroid_frequency, state.k);

	assignments.iter_mut().for_each(|a| {
        *a = config.rnd.borrow_mut().gen_range(0..k);
		centroid_frequency[*a] += 1;
	});
	kmean.p_samples.chunks_exact(kmean.p_sample_dims)
		.zip(assignments.iter().cloned())
		.for_each(|(sample, assignment)| {
			centroids.iter_mut().skip(kmean.p_sample_dims * assignment)
				.zip(sample.iter().cloned())
				.for_each(|(cv, sv)|
					*cv += sv / T::from(centroid_frequency[assignment]).unwrap()
				);
		});
}