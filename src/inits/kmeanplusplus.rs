use crate::memory::*;
use crate::{KMeans, KMeansConfig, KMeansState};
use rand::distributions::weighted::WeightedIndex;
use rand::prelude::*;
use std::ops::DerefMut;
use std::simd::{LaneCount, Simd, SupportedLaneCount};

#[inline(always)]
pub fn calculate<'a, T, const LANES: usize>(kmean: &KMeans<T, LANES>, state: &mut KMeansState<T>, config: &KMeansConfig<'a, T>)
where
    T: Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: SupportedSimdArray<T, LANES>,
{
    {
        // Randomly select first centroid
        let first_idx = config.rnd.borrow_mut().gen_range(0..kmean.sample_cnt);
        state.set_centroid_from_iter(0, kmean.p_samples.iter().skip(first_idx * kmean.p_sample_dims).cloned())
    }
    for k in 1..state.k {
        // For each following centroid...
        // Calculate distances & update cluster-assignments
        kmean.update_cluster_assignments(state, Some(k));

        //NOTE: following two calculations are not what Matlab lists on documentation, but what Matlab actually implemented...
        // Calculate sum of distances per centroid
        let distsum = state.centroid_distances.iter().cloned().sum();

        // Calculate probabilities for each of the samples, to be the new centroid
        let centroid_probabilities: Vec<T> = state.centroid_distances.iter().cloned().map(|d| d / distsum).collect();
        // Use rand's WeightedIndex to randomly draw a centroid, while respecting their probabilities
        let centroid_index = WeightedIndex::new(centroid_probabilities).unwrap();
        let sampled_centroid_id = centroid_index.sample(config.rnd.borrow_mut().deref_mut());
        state.set_centroid_from_iter(k, kmean.p_samples.iter().skip(sampled_centroid_id * kmean.p_sample_dims).cloned());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[bench]
    fn init_kmeanplusplus_f32(b: &mut Bencher) { init_kmeanplusplus::<f32, 8>(b); }
    #[bench]
    fn init_kmeanplusplus_f64(b: &mut Bencher) { init_kmeanplusplus::<f64, 8>(b); }

    fn init_kmeanplusplus<T: Primitive, const LANES: usize>(b: &mut Bencher)
    where
        T: Primitive,
        LaneCount<LANES>: SupportedLaneCount,
        Simd<T, LANES>: SupportedSimdArray<T, LANES>,
    {
        let sample_cnt = 20000;
        let sample_dims = 16;
        let k = 32;

        let mut rnd = rand::rngs::StdRng::seed_from_u64(1337);
        let mut samples = vec![T::zero(); sample_cnt * sample_dims];
        samples.iter_mut().for_each(|v| *v = rnd.gen_range(T::zero()..T::one()));
        let kmean: KMeans<_, LANES> = KMeans::new(samples, sample_cnt, sample_dims);
        let mut state = KMeansState::new::<LANES>(sample_cnt, kmean.p_sample_dims, k);
        let conf = KMeansConfig::build().random_generator(rnd).build();

        b.iter(|| {
            KMeans::init_kmeanplusplus(&kmean, &mut state, &conf);
            state.distsum
        });
    }
}
