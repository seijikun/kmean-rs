use crate::{KMeans, KMeansState, KMeansConfig, memory::*};
use rand::prelude::*;
use packed_simd::{Simd, SimdArray};

#[inline(always)] pub fn calculate<'a, T: Primitive>(kmean: &KMeans<T>, state: &mut KMeansState<T>, config: &KMeansConfig<'a, T>)
				where T: Primitive, [T;LANES]: SimdArray, Simd<[T;LANES]>: SimdWrapper<T> {

	let (assignments, centroids, centroid_frequency, k) =
		(&mut state.assignments, &mut state.centroids, &mut state.centroid_frequency, state.k);

	assignments.iter_mut().for_each(|a| {
		*a = config.rnd.borrow_mut().gen_range(0, k);
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