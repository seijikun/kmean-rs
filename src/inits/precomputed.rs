use crate::api::DistanceFunction;
use crate::memory::*;
use crate::{KMeans, KMeansConfig, KMeansState};
use std::simd::{LaneCount, Simd, SupportedLaneCount};

#[inline(always)]
pub fn calculate<T, const LANES: usize, D>(
    kmean: &KMeans<T, LANES, D>, state: &mut KMeansState<T>, _config: &KMeansConfig<'_, T>, computed: &[T],
) where
    T: Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: SupportedSimdArray<T, LANES>,
    D: DistanceFunction<T, LANES>,
{
    state.centroids = StrideBuffer::from_slice::<LANES>(kmean.sample_dims, computed);
    assert_eq!(state.centroids.centroid_cnt, state.k);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EuclideanDistance;

    #[test]
    fn train_with_precomputed_centroids() {
        let samples = vec![0.0, 1.0, 10.0, 11.0, 20.0, 21.0];
        let centroids = vec![0.0, 10.0, 20.0];
        let (sample_cnt, sample_dims) = (samples.len(), 1);

        let kmean: KMeans<f32, 8, _> = KMeans::new(samples, sample_cnt, sample_dims, EuclideanDistance);
        let result = kmean.kmeans_lloyd(3, 200, KMeans::init_precomputed(centroids), &KMeansConfig::default());

        assert_eq!(result.centroids.to_vec(), vec![0.5, 10.5, 20.5]);
    }
}
