use crate::memory::*;
use crate::{KMeans, KMeansConfig, KMeansState};
use rand::prelude::*;
use rayon::prelude::*;
use std::ops::{DerefMut, Range};
use std::simd::{LaneCount, Simd, SimdFloat, SupportedLaneCount};

struct BatchInfo {
    start_idx: usize,
    batch_size: usize,
}
impl BatchInfo {
    fn gen_range(&self, stride: usize) -> Range<usize> {
        Range {
            start: (self.start_idx * stride),
            end: (self.start_idx * stride + self.batch_size * stride),
        }
    }
}

pub(crate) struct Minibatch<T, const LANES: usize>
where
    T: Primitive,
    LaneCount<LANES>: SupportedLaneCount,
{
    _p: std::marker::PhantomData<T>,
}
impl<T, const LANES: usize> Minibatch<T, LANES>
where
    T: Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: SupportedSimdArray<T, LANES>,
{
    fn update_cluster_assignments(
        data: &KMeans<T, LANES>, state: &mut KMeansState<T>, batch: &BatchInfo, shuffled_samples: &[T], limit_k: Option<usize>,
    ) {
        let centroids = &state.centroids;
        let k = limit_k.unwrap_or(state.k);

        // TODO: Switch to par_chunks_exact, when that is merged in rayon (https://github.com/rayon-rs/rayon/pull/629).
        // par_chunks() works, because sample-dimensions are manually padded, so that there is no remainder

        // manually calculate work-packet size, because rayon does not do static scheduling (which is more apropriate here)
        let work_packet_size = batch.batch_size / rayon::current_num_threads();
        shuffled_samples[batch.gen_range(data.p_sample_dims)]
            .par_chunks_exact(data.p_sample_dims)
            .with_min_len(work_packet_size)
            .zip(state.assignments[batch.gen_range(1)].par_iter_mut())
            .zip(state.centroid_distances[batch.gen_range(1)].par_iter_mut())
            .for_each(|((s, assignment), centroid_dist)| {
                let (best_idx, best_dist) = centroids
                    .chunks_exact(data.p_sample_dims)
                    .take(k)
                    .map(|c| {
                        s.chunks_exact(LANES)
                            .map(|i| Simd::from_slice(i))
                            .zip(c.chunks_exact(LANES).map(|i| Simd::from_slice(i)))
                            .map(|(sp, cp)| sp - cp) // <sample> - <centroid>
                            .map(|v| v * v) // <vec_components> ^2
                            .sum::<Simd<T, LANES>>() // sum(<vec_components>^2)
                            .reduce_sum() // sum(sum(<vec_components>^2))
                    })
                    .enumerate()
                    .min_by(|(_, d0), (_, d1)| d0.partial_cmp(d1).unwrap())
                    .unwrap();
                *assignment = best_idx;
                *centroid_dist = best_dist;
            });
    }

    fn update_centroids(data: &KMeans<T, LANES>, state: &mut KMeansState<T>, batch: &BatchInfo, shuffled_samples: &[T]) {
        let centroid_frequency = &mut state.centroid_frequency;
        let centroids = &mut state.centroids;
        let assignments = &state.assignments;

        shuffled_samples[batch.gen_range(data.p_sample_dims)]
            .chunks_exact(data.p_sample_dims)
            .zip(assignments[batch.gen_range(1)].iter().cloned())
            .for_each(|(sample, assignment)| {
                centroid_frequency[assignment] += 1;
                let learn_rate = T::one() / T::from(centroid_frequency[assignment]).unwrap();
                let inv_learn_rate = T::one() - learn_rate;
                centroids
                    .iter_mut()
                    .skip(assignment * data.p_sample_dims)
                    .take(data.p_sample_dims)
                    .zip(sample.iter().cloned())
                    .for_each(|(c, s)| {
                        *c = inv_learn_rate * *c + learn_rate * s;
                    });
            });
    }

    fn shuffle_samples(data: &KMeans<T, LANES>, config: &KMeansConfig<'_, T>) -> (Vec<usize>, Vec<T>) {
        let mut idxs: Vec<usize> = (0..data.sample_cnt).collect();
        idxs.shuffle(config.rnd.borrow_mut().deref_mut());

        let mut shuffled_samples = AlignedFloatVec::<LANES>::create_uninitialized(data.p_samples.len());
        shuffled_samples
            .chunks_exact_mut(data.p_sample_dims)
            .zip(
                idxs.iter()
                    .map(|i| &data.p_samples[(i * data.p_sample_dims)..(i * data.p_sample_dims) + data.p_sample_dims]),
            )
            .for_each(|(dst, src)| {
                dst.copy_from_slice(src);
            });

        (idxs, shuffled_samples)
    }

    fn unshuffle_state(shuffle_idxs: &[usize], state: &mut KMeansState<T>) {
        for (from, to) in shuffle_idxs.iter().cloned().enumerate() {
            state.assignments.swap(from, to);
            state.centroid_distances.swap(from, to);
        }
    }

    #[inline(always)]
    pub fn calculate<F>(
        data: &KMeans<T, LANES>, batch_size: usize, k: usize, max_iter: usize, init: F, config: &KMeansConfig<'_, T>,
    ) -> KMeansState<T>
    where
        for<'c> F: FnOnce(&KMeans<T, LANES>, &mut KMeansState<T>, &KMeansConfig<'c, T>),
    {
        assert!(k <= data.sample_cnt);
        assert!(batch_size <= data.sample_cnt);

        // Copy and shuffle sample_data, then only take consecutive blocks (with batch_size) from there
        let (shuffle_idxs, shuffled_samples) = Self::shuffle_samples(data, config);

        let mut state = KMeansState::new::<LANES>(data.sample_cnt, data.p_sample_dims, k);
        state.distsum = T::infinity();

        // Initialize clusters and notify subscriber
        init(data, &mut state, config);
        (config.init_done)(&state);
        let mut abort_strategy = config.abort_strategy.create_logic();

        // Update cluster assignments for all samples, to get rid of the INFINITES in centroid_distances
        Self::update_cluster_assignments(
            data,
            &mut state,
            &BatchInfo {
                start_idx: 0,
                batch_size: data.sample_cnt,
            },
            &shuffled_samples,
            None,
        );

        for i in 1..=max_iter {
            // Only shuffle a beginning index for a consecutive block within the shuffled samples as batch
            let batch = BatchInfo {
                batch_size,
                start_idx: config.rnd.borrow_mut().gen_range(0..data.sample_cnt - batch_size),
            };

            Self::update_cluster_assignments(data, &mut state, &batch, &shuffled_samples, None);
            let new_distsum = state.centroid_distances.iter().cloned().sum();
            Self::update_centroids(data, &mut state, &batch, &shuffled_samples);

            // Notify subscriber about finished iteration
            (config.iteration_done)(&state, i, new_distsum);
            if !abort_strategy.next(new_distsum) {
                break;
            }
            state.distsum = new_distsum;
        }

        // Unshuffle state, so that all sample - x mappings are aligned to the original data.p_samples input
        Self::unshuffle_state(&shuffle_idxs, &mut state);
        data.update_cluster_assignments(&mut state, None);

        let (assignments, centroid_frequency, centroid_distances, distsum) = (
            &state.assignments, &mut state.centroid_frequency, &mut state.centroid_distances, &mut state.distsum,
        );
        let mut non_empty_clusters = state.k;
        rayon::scope(|s| {
            s.spawn(|_| {
                non_empty_clusters -= data.update_cluster_frequencies(assignments, centroid_frequency);
            });
            s.spawn(|_| {
                *distsum = centroid_distances.iter().cloned().sum();
            });
        });
        state.remove_padding(data.sample_dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::testing::{assert_kmeans_result_eq, KMeansShouldResult};
    use crate::AbortStrategy;

    #[test]
    fn iris_dataset_f64() {
        let samples = vec![
            1.4f64, 0.2, 1.4, 0.2, 1.3, 0.2, 1.5, 0.2, 1.4, 0.2, 1.7, 0.4, 1.4, 0.3, 1.5, 0.2, 1.4, 0.2, 1.5, 0.1, 1.5, 0.2, 1.6, 0.2, 1.4,
            0.1, 1.1, 0.1, 1.2, 0.2, 1.5, 0.4, 1.3, 0.4, 1.4, 0.3, 1.7, 0.3, 1.5, 0.3, 1.7, 0.2, 1.5, 0.4, 1.0, 0.2, 1.7, 0.5, 1.9, 0.2,
            1.6, 0.2, 1.6, 0.4, 1.5, 0.2, 1.4, 0.2, 1.6, 0.2, 1.6, 0.2, 1.5, 0.4, 1.5, 0.1, 1.4, 0.2, 1.5, 0.2, 1.2, 0.2, 1.3, 0.2, 1.4,
            0.1, 1.3, 0.2, 1.5, 0.2, 1.3, 0.3, 1.3, 0.3, 1.3, 0.2, 1.6, 0.6, 1.9, 0.4, 1.4, 0.3, 1.6, 0.2, 1.4, 0.2, 1.5, 0.2, 1.4, 0.2,
            4.7, 1.4, 4.5, 1.5, 4.9, 1.5, 4.0, 1.3, 4.6, 1.5, 4.5, 1.3, 4.7, 1.6, 3.3, 1.0, 4.6, 1.3, 3.9, 1.4, 3.5, 1.0, 4.2, 1.5, 4.0,
            1.0, 4.7, 1.4, 3.6, 1.3, 4.4, 1.4, 4.5, 1.5, 4.1, 1.0, 4.5, 1.5, 3.9, 1.1, 4.8, 1.8, 4.0, 1.3, 4.9, 1.5, 4.7, 1.2, 4.3, 1.3,
            4.4, 1.4, 4.8, 1.4, 5.0, 1.7, 4.5, 1.5, 3.5, 1.0, 3.8, 1.1, 3.7, 1.0, 3.9, 1.2, 5.1, 1.6, 4.5, 1.5, 4.5, 1.6, 4.7, 1.5, 4.4,
            1.3, 4.1, 1.3, 4.0, 1.3, 4.4, 1.2, 4.6, 1.4, 4.0, 1.2, 3.3, 1.0, 4.2, 1.3, 4.2, 1.2, 4.2, 1.3, 4.3, 1.3, 3.0, 1.1, 4.1, 1.3,
            6.0, 2.5, 5.1, 1.9, 5.9, 2.1, 5.6, 1.8, 5.8, 2.2, 6.6, 2.1, 4.5, 1.7, 6.3, 1.8, 5.8, 1.8, 6.1, 2.5, 5.1, 2.0, 5.3, 1.9, 5.5,
            2.1, 5.0, 2.0, 5.1, 2.4, 5.3, 2.3, 5.5, 1.8, 6.7, 2.2, 6.9, 2.3, 5.0, 1.5, 5.7, 2.3, 4.9, 2.0, 6.7, 2.0, 4.9, 1.8, 5.7, 2.1,
            6.0, 1.8, 4.8, 1.8, 4.9, 1.8, 5.6, 2.1, 5.8, 1.6, 6.1, 1.9, 6.4, 2.0, 5.6, 2.2, 5.1, 1.5, 5.6, 1.4, 6.1, 2.3, 5.6, 2.4, 5.5,
            1.8, 4.8, 1.8, 5.4, 2.1, 5.6, 2.4, 5.1, 2.3, 5.1, 1.9, 5.9, 2.3, 5.7, 2.5, 5.2, 2.3, 5.0, 1.9, 5.2, 2.0, 5.4, 2.3, 5.1, 1.8,
        ];

        let kmean: KMeans<f64, 8> = KMeans::new(samples, 150, 2);
        let rnd = rand::rngs::StdRng::seed_from_u64(3);
        let conf = KMeansConfig::build()
            .random_generator(rnd)
            .abort_strategy(AbortStrategy::NoImprovementForXIterations {
                x: 5,
                threshold: 0.0005f64,
                abort_on_negative: true,
            })
            .build();
        let res = kmean.kmeans_minibatch(30, 3, 100, KMeans::init_kmeanplusplus, &conf);

        // SHOULD solution
        let should = KMeansShouldResult {
            distsum: 31.691483430123924,
            sample_dims: 2,
            assignments: vec![
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2,
                0, 2, 2, 0, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            ],
            centroid_distances: vec![
                0.008367214532872126, 0.008367214532872126, 0.03307309688581347, 0.0036613321799307407, 0.008367214532872126,
                0.07248486159169505, 0.00748486159169566, 0.0036613321799307407, 0.008367214532872126, 0.02454368512110721,
                0.0036613321799307407, 0.018955449826989393, 0.029249567474048596, 0.16336721453287262, 0.07777897923875488,
                0.021896626297577815, 0.051308391003460546, 0.00748486159169566, 0.05336721453287151, 0.0027789792387542737,
                0.05424956747404797, 0.021896626297577815, 0.2271907439446376, 0.11160250865051857, 0.18483780276816517,
                0.018955449826989393, 0.037190743944636465, 0.0036613321799307407, 0.008367214532872126, 0.018955449826989393,
                0.018955449826989393, 0.021896626297577815, 0.02454368512110721, 0.008367214532872126, 0.0036613321799307407,
                0.07777897923875488, 0.03307309688581347, 0.029249567474048596, 0.03307309688581347, 0.0036613321799307407,
                0.032190743944637, 0.032190743944637, 0.03307309688581347, 0.1354260380622835, 0.20307309688581224, 0.00748486159169566,
                0.018955449826989393, 0.008367214532872126, 0.0036613321799307407, 0.008367214532872126, 0.11522168220591258,
                0.02902449910732018, 0.30057379488196956, 0.1404329498115445, 0.06691182305098226, 0.029869569529855745,
                0.15437661178337703, 1.2864892878397125, 0.06775689347351782, 0.21212309065661458, 0.9022639357270366,
                0.035362527276333384, 0.2917005554453479, 0.11522168220591258, 0.5888836540368955, 0.0015597103749257503,
                0.02902449910732018, 0.22958787938901032, 0.02902449910732018, 0.30339069629041787, 0.3514188653045034, 0.1404329498115445,
                0.30057379488196956, 0.15606675262844816, 0.014094921642531275, 0.0015597103749257503, 0.19310900614957455,
                0.4976160484030958, 0.02902449910732018, 0.9022639357270366, 0.40550337234675576, 0.5980385836143609, 0.25296816107915016,
                0.5381776147959252, 0.02902449910732018, 0.05860196389605243, 0.1247991469946448, 0.011982245586193527,
                0.07832027375520695, 0.1404329498115445, 0.04240478079746134, 0.05733435826225004, 0.17085548502281234, 1.2864892878397125,
                0.03620759769886895, 0.06663013291013677, 0.03620759769886895, 0.014094921642531275, 1.9424047807974576,
                0.07832027375520695, 0.2971061862244855, 0.34371332908163893, 0.058356186224487565, 0.0787133290816339,
                0.03549904336734527, 0.88460618622448, 0.10817942868478458, 0.48496332908162676, 0.09478475765306044, 0.37514190051019847,
                0.3188919005102101, 0.1597847576530653, 0.026213329081634155, 0.4408561862244964, 0.4196061862244948, 0.18049904336734995,
                0.10067761479592055, 1.0978204719387654, 1.5890704719387636, 0.4184611188256314, 0.05264190051020309, 0.5828204719387825,
                1.0874633290816231, 0.44930618924816623, 0.002284757653060791, 0.1908561862244871, 0.3514188653045034, 0.44930618924816623,
                0.0042490433673475075, 0.24442761479591799, 0.22407047193877133, 0.5533561862244832, 0.01942761479591874,
                0.5563484427692932, 0.45799904336734926, 0.244784757653056, 0.10978475765306102, 0.10067761479592055, 0.3514188653045034,
                0.06817761479592066, 0.10978475765306102, 0.3644276147959236, 0.34371332908163893, 0.10871332908162987,
                0.18299904336734557, 0.2624633290816364, 0.46567761479592523, 0.21692761479592287, 0.11853475765306296,
                0.38853475765306766,
            ],
            centroids: vec![
                4.360563380281689, 1.402112676056339, 1.4735294117647069, 0.25441176470588234, 5.659821428571433, 2.074107142857144,
            ],
        };

        assert_eq!(res.sample_dims, 8);
        assert_kmeans_result_eq(should, res);
    }

    #[test]
    fn iris_dataset_f32() {
        let samples = vec![
            1.4f32, 0.2, 1.4, 0.2, 1.3, 0.2, 1.5, 0.2, 1.4, 0.2, 1.7, 0.4, 1.4, 0.3, 1.5, 0.2, 1.4, 0.2, 1.5, 0.1, 1.5, 0.2, 1.6, 0.2, 1.4,
            0.1, 1.1, 0.1, 1.2, 0.2, 1.5, 0.4, 1.3, 0.4, 1.4, 0.3, 1.7, 0.3, 1.5, 0.3, 1.7, 0.2, 1.5, 0.4, 1.0, 0.2, 1.7, 0.5, 1.9, 0.2,
            1.6, 0.2, 1.6, 0.4, 1.5, 0.2, 1.4, 0.2, 1.6, 0.2, 1.6, 0.2, 1.5, 0.4, 1.5, 0.1, 1.4, 0.2, 1.5, 0.2, 1.2, 0.2, 1.3, 0.2, 1.4,
            0.1, 1.3, 0.2, 1.5, 0.2, 1.3, 0.3, 1.3, 0.3, 1.3, 0.2, 1.6, 0.6, 1.9, 0.4, 1.4, 0.3, 1.6, 0.2, 1.4, 0.2, 1.5, 0.2, 1.4, 0.2,
            4.7, 1.4, 4.5, 1.5, 4.9, 1.5, 4.0, 1.3, 4.6, 1.5, 4.5, 1.3, 4.7, 1.6, 3.3, 1.0, 4.6, 1.3, 3.9, 1.4, 3.5, 1.0, 4.2, 1.5, 4.0,
            1.0, 4.7, 1.4, 3.6, 1.3, 4.4, 1.4, 4.5, 1.5, 4.1, 1.0, 4.5, 1.5, 3.9, 1.1, 4.8, 1.8, 4.0, 1.3, 4.9, 1.5, 4.7, 1.2, 4.3, 1.3,
            4.4, 1.4, 4.8, 1.4, 5.0, 1.7, 4.5, 1.5, 3.5, 1.0, 3.8, 1.1, 3.7, 1.0, 3.9, 1.2, 5.1, 1.6, 4.5, 1.5, 4.5, 1.6, 4.7, 1.5, 4.4,
            1.3, 4.1, 1.3, 4.0, 1.3, 4.4, 1.2, 4.6, 1.4, 4.0, 1.2, 3.3, 1.0, 4.2, 1.3, 4.2, 1.2, 4.2, 1.3, 4.3, 1.3, 3.0, 1.1, 4.1, 1.3,
            6.0, 2.5, 5.1, 1.9, 5.9, 2.1, 5.6, 1.8, 5.8, 2.2, 6.6, 2.1, 4.5, 1.7, 6.3, 1.8, 5.8, 1.8, 6.1, 2.5, 5.1, 2.0, 5.3, 1.9, 5.5,
            2.1, 5.0, 2.0, 5.1, 2.4, 5.3, 2.3, 5.5, 1.8, 6.7, 2.2, 6.9, 2.3, 5.0, 1.5, 5.7, 2.3, 4.9, 2.0, 6.7, 2.0, 4.9, 1.8, 5.7, 2.1,
            6.0, 1.8, 4.8, 1.8, 4.9, 1.8, 5.6, 2.1, 5.8, 1.6, 6.1, 1.9, 6.4, 2.0, 5.6, 2.2, 5.1, 1.5, 5.6, 1.4, 6.1, 2.3, 5.6, 2.4, 5.5,
            1.8, 4.8, 1.8, 5.4, 2.1, 5.6, 2.4, 5.1, 2.3, 5.1, 1.9, 5.9, 2.3, 5.7, 2.5, 5.2, 2.3, 5.0, 1.9, 5.2, 2.0, 5.4, 2.3, 5.1, 1.8,
        ];

        let kmean: KMeans<f32, 8> = KMeans::new(samples, 150, 2);
        let rnd = rand::rngs::StdRng::seed_from_u64(3);
        let conf = KMeansConfig::build()
            .random_generator(rnd)
            .abort_strategy(AbortStrategy::NoImprovementForXIterations {
                x: 5,
                threshold: 0.0005f32,
                abort_on_negative: true,
            })
            .build();
        let res = kmean.kmeans_minibatch(30, 3, 100, KMeans::init_kmeanplusplus, &conf);

        // SHOULD solution
        let should = KMeansShouldResult {
            distsum: 31.751724,
            sample_dims: 2,
            assignments: vec![
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
                0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2,
                2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            ],
            centroid_distances: vec![
                0.01267357, 0.01267357, 0.037013218, 0.0083339345, 0.01267357, 0.06493769, 0.005315068, 0.0083339345, 0.01267357,
                0.03569244, 0.0083339345, 0.023994308, 0.040032074, 0.17305095, 0.08135281, 0.01361693, 0.042296212, 0.005315068,
                0.052296188, 0.0009754318, 0.05965469, 0.01361693, 0.2300321, 0.09757918, 0.19097538, 0.023994308, 0.029277302,
                0.0083339345, 0.01267357, 0.023994308, 0.023994308, 0.01361693, 0.03569244, 0.01267357, 0.0083339345, 0.08135281,
                0.037013218, 0.040032074, 0.037013218, 0.0083339345, 0.029654715, 0.029654715, 0.037013218, 0.114560306, 0.19625838,
                0.005315068, 0.023994308, 0.01267357, 0.0083339345, 0.01267357, 0.16893126, 0.075018674, 0.3967574, 0.08893235, 0.12545326,
                0.041105583, 0.24284437, 1.0950203, 0.09154017, 0.16545418, 0.73588943, 0.043714777, 0.1880627, 0.16893126, 0.48719388,
                0.0176275, 0.075018674, 0.13849738, 0.075018674, 0.20458452, 0.48719242, 0.08893235, 0.3967574, 0.17501816, 0.00023629011,
                0.0176275, 0.25936627, 0.39401728, 0.075018674, 0.73588943, 0.29415, 0.4567587, 0.17154106, 0.3755859, 0.075018674,
                0.12197524, 0.1958878, 0.010670955, 0.039367035, 0.08893235, 0.023714393, 0.09849672, 0.10197579, 1.0950203, 0.009801679,
                0.02284512, 0.009801679, 0.00023629011, 1.7306727, 0.039367035, 0.4355868, 0.20323333, 0.1412727, 0.061272323, 0.09970424,
                1.1500959, 0.18893182, 0.6500962, 0.12950772, 0.5397043, 0.18578245, 0.07146844, 0.0048020603, 0.28166473, 0.31597906,
                0.12166494, 0.057154696, 1.3967625, 1.9475476, 0.5271919, 0.09813553, 0.39754698, 1.3716642, 0.45244873, 0.03303728,
                0.27774292, 0.48719242, 0.45244873, 0.008919688, 0.26440942, 0.34440956, 0.7593119, 0.03146885, 0.4730368, 0.4110758,
                0.39460605, 0.13656716, 0.057154696, 0.48719242, 0.020684395, 0.13656716, 0.25342983, 0.20323333, 0.20637095, 0.24323381,
                0.17754751, 0.2991156, 0.10990014, 0.08578265, 0.24068421,
            ],
            centroids: vec![4.297827, 1.3152173, 1.4716982, 0.28679252, 5.529412, 2.0372543],
        };

        assert_eq!(res.sample_dims, 8);
        assert_kmeans_result_eq(should, res);
    }
}
