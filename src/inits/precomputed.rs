use crate::memory::*;
use crate::{KMeans, KMeansConfig, KMeansState};
use std::simd::{LaneCount, Simd, SupportedLaneCount};

#[inline(always)]
pub fn calculate<T, const LANES: usize>(
    kmean: &KMeans<T, LANES>, state: &mut KMeansState<T>, _config: &KMeansConfig<'_, T>, computed: Vec<T>,
) where
    T: Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: SupportedSimdArray<T, LANES>,
{
    computed.chunks_exact(kmean.p_sample_dims).enumerate().for_each(|(ci, c)| {
        if ci > state.k {
            panic!("Initialized with more centroids than k");
        }
        state.set_centroid_from_iter(ci, c.iter().cloned());
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_with_precomputed_centroids() {
        let samples = vec![0.0, 1.0, 10.0, 11.0, 20.0, 21.0];
        let centroids = vec![0.0, 21.0];
        let (sample_cnt, sample_dims) = (samples.len(), 1);

        let kmean: KMeans<f32, 8> = KMeans::new(samples, sample_cnt, sample_dims);
        let result = kmean.kmeans_lloyd(2, 200, KMeans::init_precomputed(centroids), &KMeansConfig::default());

        assert_eq!(result.centroids, vec![5.5, 20.5]);
    }
}
