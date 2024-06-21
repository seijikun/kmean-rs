use crate::memory::*;
use crate::{KMeans, KMeansConfig, KMeansState};
use rand::prelude::*;
use std::ops::DerefMut;
use std::simd::{LaneCount, Simd, SupportedLaneCount};

#[inline(always)]
pub fn calculate<T, const LANES: usize>(kmean: &KMeans<T, LANES>, state: &mut KMeansState<T>, config: &KMeansConfig<'_, T>)
where
    T: Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: SupportedSimdArray<T, LANES>,
{
    kmean
        .p_samples
        .chunks_exact(kmean.p_sample_dims)
        .choose_multiple(config.rnd.borrow_mut().deref_mut(), state.k)
        .iter()
        .cloned()
        .enumerate()
        .for_each(|(ci, c)| {
            // Copy randomly chosen centroids into state.centroids
            state.set_centroid_from_iter(ci, c.iter().cloned());
        });
}
