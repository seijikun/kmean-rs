use crate::{KMeans, KMeansState, memory::*};
use rand::prelude::*;
use packed_simd::{Simd, SimdArray};

#[inline(always)] pub fn calculate<'a, T: Primitive>(kmean: &KMeans<T>, state: &mut KMeansState<T>, rnd: &'a mut dyn RngCore)
				where T: Primitive, [T;LANES]: SimdArray, Simd<[T;LANES]>: SimdWrapper<T> {
	kmean.p_samples.chunks_exact(kmean.p_sample_dims)
		.choose_multiple(rnd, state.k).iter().cloned()
		.enumerate()
		.for_each(|(ci, c)| { // Copy randomly chosen centroids into state.centroids
			state.set_centroid_from_iter(ci, c.iter().cloned());
		});
}