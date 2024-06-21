use crate::{KMeans, KMeansState, KMeansConfig, memory::*};
use rand::prelude::*;
use std::simd::{LaneCount, Simd, SupportedLaneCount};
use std::ops::DerefMut;

#[inline(always)]
pub fn calculate<'a, T, const LANES: usize>(kmean: &KMeans<T, LANES>, state: &mut KMeansState<T>, config: &KMeansConfig<'a, T>)
where
    T: Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: SupportedSimdArray<T, LANES>
{
    kmean.p_samples.chunks_exact(kmean.p_sample_dims)
		.choose_multiple(config.rnd.borrow_mut().deref_mut(), state.k).iter().cloned()
		.enumerate()
		.for_each(|(ci, c)| { // Copy randomly chosen centroids into state.centroids
			state.set_centroid_from_iter(ci, c.iter().cloned());
		});
}