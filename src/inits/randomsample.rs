use crate::{KMeans, KMeansState, KMeansConfig, memory::*};
use rand::prelude::*;
use std::ops::DerefMut;
use packed_simd::{Simd, SimdArray};

#[inline(always)] pub fn calculate<'a, T: Primitive>(kmean: &KMeans<T>, state: &mut KMeansState<T>, config: &KMeansConfig<'a, T>)
				where T: Primitive, [T;LANES]: SimdArray, Simd<[T;LANES]>: SimdWrapper<T> {
	kmean.p_samples.chunks_exact(kmean.p_sample_dims)
		.choose_multiple(config.rnd.borrow_mut().deref_mut(), state.k).iter().cloned()
		.enumerate()
		.for_each(|(ci, c)| { // Copy randomly chosen centroids into state.centroids
			state.set_centroid_from_iter(ci, c.iter().cloned());
		});
}