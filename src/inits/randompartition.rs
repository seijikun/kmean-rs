use crate::memory::*;
use crate::{KMeans, KMeansConfig, KMeansState};
use rand::prelude::*;
use std::simd::{LaneCount, Simd, SupportedLaneCount};

#[inline(always)]
pub fn calculate<'a, T, const LANES: usize>(kmean: &KMeans<T, LANES>, state: &mut KMeansState<T>, config: &KMeansConfig<'a, T>)
where
    T: Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: SupportedSimdArray<T, LANES>,
{
    let (assignments, centroids, centroid_frequency, k) =
        (&mut state.assignments, &mut state.centroids, &mut state.centroid_frequency, state.k);

    assignments.iter_mut().for_each(|a| {
        *a = config.rnd.borrow_mut().gen_range(0..k);
        centroid_frequency[*a] += 1;
    });
    kmean
        .p_samples
        .chunks_exact(kmean.p_sample_dims)
        .zip(assignments.iter().cloned())
        .for_each(|(sample, assignment)| {
            centroids
                .iter_mut()
                .skip(kmean.p_sample_dims * assignment)
                .zip(sample.iter().cloned())
                .for_each(|(cv, sv)| *cv += sv / T::from(centroid_frequency[assignment]).unwrap());
        });
}
