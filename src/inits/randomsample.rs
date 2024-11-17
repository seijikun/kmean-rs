use crate::api::DistanceFunction;
use crate::memory::*;
use crate::{KMeans, KMeansConfig, KMeansState};
use rand::prelude::*;
use std::ops::DerefMut;
use std::simd::{LaneCount, Simd, SupportedLaneCount};

#[inline(always)]
pub fn calculate<T, const LANES: usize, D>(kmean: &KMeans<T, LANES, D>, state: &mut KMeansState<T>, config: &KMeansConfig<'_, T>)
where
    T: Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: SupportedSimdArray<T, LANES>,
    D: DistanceFunction<T, LANES>,
{
    kmean
        .p_samples
        .chunks_exact_stride()
        .choose_multiple(config.rnd.borrow_mut().deref_mut(), state.k)
        .iter()
        .cloned()
        .enumerate()
        .for_each(|(ci, c)| {
            // Copy randomly chosen centroids into state.centroids
            state.centroids.set_nth_from_iter(ci, c.iter().cloned());
        });
}
