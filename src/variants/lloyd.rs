use crate::{KMeans, KMeansState, KMeansConfig, memory::*};
use packed_simd::{Simd, SimdArray};

pub(crate) struct Lloyd<T> where T: Primitive, [T;LANES]: SimdArray, Simd<[T;LANES]>: SimdWrapper<T>{
	_p: std::marker::PhantomData<T>
}
impl<T> Lloyd<T> where T: Primitive, [T;LANES]: SimdArray, Simd<[T;LANES]>: SimdWrapper<T> {
    fn update_centroids(data: &KMeans<T>, state: &mut KMeansState<T>) -> T {
        let chunks_per_sample = data.p_sample_dims / LANES;
        // Sum all samples in a cluster together into new_centroids
        // Count non-empty clusters
        let mut used_centroids_cnt = 0;
        let mut new_centroids = AlignedFloatVec::new(state.centroids.len());
        let mut new_distsum = T::zero();

        let (centroid_frequency, assignments, centroid_distances) = (&mut state.centroid_frequency, &state.assignments, &state.centroid_distances);
        rayon::scope(|s| {
            s.spawn(|_| {
				used_centroids_cnt = data.update_cluster_frequencies(assignments, centroid_frequency);
            });
            s.spawn(|_| {
                data.p_samples.chunks_exact(data.p_sample_dims)
                    .zip(assignments.iter().cloned())
                    .for_each(|(s, centroid_id)| {
                        new_centroids.chunks_exact_mut(LANES).skip(centroid_id * chunks_per_sample).take(chunks_per_sample)
                            .zip(s.chunks_exact(LANES).map(|i| unsafe { Simd::<[T;LANES]>::from_slice_aligned_unchecked(i) }))
                            .for_each(|(c,s)| unsafe { // For each chunk
                                (Simd::<[T;LANES]>::from_slice_aligned_unchecked(c) + s).write_to_slice_aligned_unchecked(c);
                            });
                    });
            });
            s.spawn(|_| {
                new_distsum = centroid_distances.iter().cloned().sum();
            });
        });

        // Use used_centroids_cnt variable to check, whether there are empty clusters
        // When there are, assign bad samples to empty clusters
        if used_centroids_cnt != state.k {
            let mut distance_sorted_samples: Vec<usize> = (0..data.sample_cnt).collect();
            distance_sorted_samples.sort_unstable_by(
                |&i1, &i2| state.centroid_distances[i1].partial_cmp(&state.centroid_distances[i2]).unwrap());

            // Assign empty clusters
            for i in 0..state.k {
                if state.centroid_frequency[i] == 0 {
                    // Find the sample with the highest distance to its centroid, that is not alone in its cluster
                    let mut sample_id = std::usize::MAX;
                    let mut prev_centroid_id = std::usize::MAX;
                    for j in (0..data.sample_cnt).rev() {
                        sample_id = distance_sorted_samples[j];
                        prev_centroid_id = state.assignments[sample_id];
                        if state.centroid_frequency[prev_centroid_id] > 1 {
                            break;
                        }
                    }
                    assert!(sample_id != std::usize::MAX && prev_centroid_id != std::usize::MAX);
                    // Re-Assign found sample to centroid without any samples
                    state.centroid_frequency[prev_centroid_id] -= 1;
                    state.centroid_frequency[i] += 1;
                    new_distsum -= state.centroid_distances[sample_id];
                    // Centroid is moved into the chosen point -> the points centroid distance is 0
                    state.centroid_distances[sample_id] = T::zero();
                    // new_centroids is a sum of all points within a centroid here.
                    // Subtract chosen sample from its previous centroid
                    new_centroids.iter_mut().skip(prev_centroid_id * data.p_sample_dims).take(data.p_sample_dims)
                        .zip(data.p_samples.iter().skip(sample_id * data.p_sample_dims).cloned())
                        .for_each(|(cv,sv)| { *cv -= sv; });
                    // Chosen sample is single point in cluster -> set cluster's sum to chosen point
                    new_centroids.iter_mut().skip(i * data.p_sample_dims).take(data.p_sample_dims)
                        .zip(data.p_samples.iter().skip(sample_id * data.p_sample_dims).cloned())
                        .for_each(|(cv,sv)| { *cv = sv; });
                    state.assignments[sample_id] = i;
                }
            }
        }
        // Calculate new centroids from updated cluster_assignments
        state.centroids.chunks_exact_mut(data.p_sample_dims)
            .zip(new_centroids.chunks_exact(data.p_sample_dims))
            .zip(state.centroid_frequency.iter().cloned())
            .for_each(|((c,nc),cfreq)| {
                let cfreq = Simd::<[T;LANES]>::splat(T::from(cfreq).unwrap());
                c.chunks_exact_mut(LANES)
                    .zip(nc.chunks_exact(LANES).map(|v| unsafe { Simd::<[T;LANES]>::from_slice_aligned_unchecked(v) }))
                    .for_each(|(c,nc)| unsafe {
                        (nc / cfreq).write_to_slice_aligned_unchecked(c);
                    });
            });
        new_distsum
    }

    #[inline(always)] pub fn calculate<'a, F>(data: &KMeans<T>, k: usize, max_iter: usize, init: F, config: &KMeansConfig<'a, T>) -> KMeansState<T>
                where for<'c> F: FnOnce(&KMeans<T>, &mut KMeansState<T>, &KMeansConfig<'c, T>) {
        assert!(k <= data.sample_cnt);

        let mut state = KMeansState::new(data.sample_cnt, data.p_sample_dims, k);
        state.distsum = T::infinity();

        // Initialize clusters and notify subscriber
        init(&data, &mut state, config);
        (config.init_done)(&state);
        let mut abort_strategy = config.abort_strategy.create_logic();

        for i in 1..=max_iter {
            data.update_cluster_assignments(&mut state, None);
            let new_distsum = Self::update_centroids(data, &mut state);

			// Notify subscriber about finished iteration
			(config.iteration_done)(&state, i, new_distsum);
            if !abort_strategy.next(new_distsum) {
                break;
            }
            state.distsum = new_distsum;
        }

        data.update_centroid_distances(&mut state);
        state.distsum = state.centroid_distances.iter().cloned().sum();
        state.remove_padding(data.sample_dims)
    }
}




#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn iris_dataset_f64() where {
        let samples = vec![1.4f64, 0.2, 1.4, 0.2, 1.3, 0.2, 1.5, 0.2, 1.4, 0.2, 1.7, 0.4, 1.4, 0.3, 1.5, 0.2, 1.4, 0.2, 1.5, 0.1, 1.5, 0.2, 1.6, 0.2, 1.4, 0.1, 1.1, 0.1, 1.2, 0.2, 1.5, 0.4, 1.3, 0.4, 1.4, 0.3, 1.7, 0.3, 1.5, 0.3, 1.7, 0.2, 1.5, 0.4, 1.0, 0.2, 1.7, 0.5, 1.9, 0.2, 1.6, 0.2, 1.6, 0.4, 1.5, 0.2, 1.4, 0.2, 1.6, 0.2, 1.6, 0.2, 1.5, 0.4, 1.5, 0.1, 1.4, 0.2, 1.5, 0.2, 1.2, 0.2, 1.3, 0.2, 1.4, 0.1, 1.3, 0.2, 1.5, 0.2, 1.3, 0.3, 1.3, 0.3, 1.3, 0.2, 1.6, 0.6, 1.9, 0.4, 1.4, 0.3, 1.6, 0.2, 1.4, 0.2, 1.5, 0.2, 1.4, 0.2, 4.7, 1.4, 4.5, 1.5, 4.9, 1.5, 4.0, 1.3, 4.6, 1.5, 4.5, 1.3, 4.7, 1.6, 3.3, 1.0, 4.6, 1.3, 3.9, 1.4, 3.5, 1.0, 4.2, 1.5, 4.0, 1.0, 4.7, 1.4, 3.6, 1.3, 4.4, 1.4, 4.5, 1.5, 4.1, 1.0, 4.5, 1.5, 3.9, 1.1, 4.8, 1.8, 4.0, 1.3, 4.9, 1.5, 4.7, 1.2, 4.3, 1.3, 4.4, 1.4, 4.8, 1.4, 5.0, 1.7, 4.5, 1.5, 3.5, 1.0, 3.8, 1.1, 3.7, 1.0, 3.9, 1.2, 5.1, 1.6, 4.5, 1.5, 4.5, 1.6, 4.7, 1.5, 4.4, 1.3, 4.1, 1.3, 4.0, 1.3, 4.4, 1.2, 4.6, 1.4, 4.0, 1.2, 3.3, 1.0, 4.2, 1.3, 4.2, 1.2, 4.2, 1.3, 4.3, 1.3, 3.0, 1.1, 4.1, 1.3, 6.0, 2.5, 5.1, 1.9, 5.9, 2.1, 5.6, 1.8, 5.8, 2.2, 6.6, 2.1, 4.5, 1.7, 6.3, 1.8, 5.8, 1.8, 6.1, 2.5, 5.1, 2.0, 5.3, 1.9, 5.5, 2.1, 5.0, 2.0, 5.1, 2.4, 5.3, 2.3, 5.5, 1.8, 6.7, 2.2, 6.9, 2.3, 5.0, 1.5, 5.7, 2.3, 4.9, 2.0, 6.7, 2.0, 4.9, 1.8, 5.7, 2.1, 6.0, 1.8, 4.8, 1.8, 4.9, 1.8, 5.6, 2.1, 5.8, 1.6, 6.1, 1.9, 6.4, 2.0, 5.6, 2.2, 5.1, 1.5, 5.6, 1.4, 6.1, 2.3, 5.6, 2.4, 5.5, 1.8, 4.8, 1.8, 5.4, 2.1, 5.6, 2.4, 5.1, 2.3, 5.1, 1.9, 5.9, 2.3, 5.7, 2.5, 5.2, 2.3, 5.0, 1.9, 5.2, 2.0, 5.4, 2.3, 5.1, 1.8];

        let kmean = KMeans::new(samples, 150, 2);
        let rnd = rand::rngs::StdRng::seed_from_u64(1);
		let conf = KMeansConfig::build().random_generator(rnd).build();
        let res = kmean.kmeans_lloyd(3, 100, KMeans::init_kmeanplusplus, &conf);

        // SHOULD solution
        let should_assignments = vec![1usize, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
        let should_centroid_distances = vec![0.005960000000000026, 0.005960000000000026, 0.028360000000000038, 0.003559999999999977, 0.005960000000000026, 0.08035999999999992, 0.006760000000000042, 0.003559999999999977, 0.005960000000000026, 0.02275999999999996, 0.003559999999999977, 0.021159999999999964, 0.02516000000000001, 0.15236000000000005, 0.07076000000000011, 0.02516000000000002, 0.049960000000000074, 0.006760000000000042, 0.0595599999999999, 0.004359999999999994, 0.05875999999999988, 0.02516000000000002, 0.21556000000000014, 0.12115999999999995, 0.19395999999999974, 0.021159999999999964, 0.042760000000000006, 0.003559999999999977, 0.005960000000000026, 0.021159999999999964, 0.021159999999999964, 0.02516000000000002, 0.02275999999999996, 0.005960000000000026, 0.003559999999999977, 0.07076000000000011, 0.028360000000000038, 0.02516000000000001, 0.028360000000000038, 0.003559999999999977, 0.029160000000000054, 0.029160000000000054, 0.028360000000000038, 0.14436000000000004, 0.2155599999999998, 0.006760000000000042, 0.021159999999999964, 0.005960000000000026, 0.003559999999999977, 0.005960000000000026, 0.18889053254437893, 0.07812130177514806, 0.4227366863905332, 0.07427514792899402, 0.134275147928994, 0.05504437869822485, 0.2519674556213022, 1.0565828402366864, 0.11119822485207079, 0.13965976331360952, 0.7088905325443784, 0.029659763313609536, 0.18965976331360923, 0.18889053254437893, 0.44965976331360924, 0.020428994082840376, 0.07812130177514806, 0.1458136094674555, 0.07812130177514806, 0.19504437869822466, 0.4911982248520712, 0.07427514792899402, 0.4227366863905332, 0.20581360946745575, 0.002736686390532506, 0.020428994082840376, 0.28504437869822474, 0.4689236111111097, 0.07812130177514806, 0.7088905325443784, 0.2788905325443786, 0.4411982248520705, 0.15658284023668634, 0.4372569444444434, 0.07812130177514806, 0.11965976331360972, 0.21042899408284055, 0.01889053254437878, 0.030428994082840305, 0.07427514792899402, 0.03735207100591719, 0.11273668639053239, 0.09273668639053242, 1.0565828402366864, 0.006582840236686325, 0.025044378698224738, 0.006582840236686325, 0.002736686390532506, 1.6696597633136092, 0.030428994082840305, 0.3772569444444456, 0.2647569444444437, 0.096423611111112, 0.056423611111110925, 0.06809027777777829, 1.0122569444444458, 0.18119822485207124, 0.5522569444444454, 0.09809027777777793, 0.4680902777777788, 0.24725694444444377, 0.10642361111111055, 0.01309027777777764, 0.35642361111110993, 0.377256944444444, 0.15642361111111072, 0.0655902777777774, 1.2455902777777805, 1.769756944444448, 0.5588905325443789, 0.07975694444444478, 0.48559027777777586, 1.2205902777777804, 0.5405902777777757, 0.014756944444444746, 0.21975694444444505, 0.4911982248520712, 0.5405902777777757, 0.003923611111111172, 0.23309027777777774, 0.27309027777777833, 0.6480902777777799, 0.02642361111111129, 0.5347569444444434, 0.40642361111111075, 0.32309027777777855, 0.1314236111111113, 0.0655902777777774, 0.4911982248520712, 0.042256944444443965, 0.1314236111111113, 0.3147569444444439, 0.2647569444444437, 0.16142361111111203, 0.22475694444444502, 0.22559027777777696, 0.37392361111110983, 0.15809027777777682, 0.107256944444444, 0.3022569444444436];
        let should_centroids = vec![4.269230769230769, 1.342307692307692, 1.4620000000000002, 0.2459999999999999, 5.595833333333332, 2.0374999999999996];
		let should_centroid_frequency = vec![52, 50, 48];

        assert_eq!(res.distsum, 31.371358974358966);
        assert_eq!(res.sample_dims, LANES);
        assert_eq!(res.assignments, should_assignments);
        assert_eq!(res.centroid_distances, should_centroid_distances);
        assert_eq!(res.centroids, should_centroids);
		assert_eq!(res.centroid_frequency, should_centroid_frequency);
    }

    #[test]
    fn iris_dataset_f32() where {
        let samples = vec![1.4f32, 0.2, 1.4, 0.2, 1.3, 0.2, 1.5, 0.2, 1.4, 0.2, 1.7, 0.4, 1.4, 0.3, 1.5, 0.2, 1.4, 0.2, 1.5, 0.1, 1.5, 0.2, 1.6, 0.2, 1.4, 0.1, 1.1, 0.1, 1.2, 0.2, 1.5, 0.4, 1.3, 0.4, 1.4, 0.3, 1.7, 0.3, 1.5, 0.3, 1.7, 0.2, 1.5, 0.4, 1.0, 0.2, 1.7, 0.5, 1.9, 0.2, 1.6, 0.2, 1.6, 0.4, 1.5, 0.2, 1.4, 0.2, 1.6, 0.2, 1.6, 0.2, 1.5, 0.4, 1.5, 0.1, 1.4, 0.2, 1.5, 0.2, 1.2, 0.2, 1.3, 0.2, 1.4, 0.1, 1.3, 0.2, 1.5, 0.2, 1.3, 0.3, 1.3, 0.3, 1.3, 0.2, 1.6, 0.6, 1.9, 0.4, 1.4, 0.3, 1.6, 0.2, 1.4, 0.2, 1.5, 0.2, 1.4, 0.2, 4.7, 1.4, 4.5, 1.5, 4.9, 1.5, 4.0, 1.3, 4.6, 1.5, 4.5, 1.3, 4.7, 1.6, 3.3, 1.0, 4.6, 1.3, 3.9, 1.4, 3.5, 1.0, 4.2, 1.5, 4.0, 1.0, 4.7, 1.4, 3.6, 1.3, 4.4, 1.4, 4.5, 1.5, 4.1, 1.0, 4.5, 1.5, 3.9, 1.1, 4.8, 1.8, 4.0, 1.3, 4.9, 1.5, 4.7, 1.2, 4.3, 1.3, 4.4, 1.4, 4.8, 1.4, 5.0, 1.7, 4.5, 1.5, 3.5, 1.0, 3.8, 1.1, 3.7, 1.0, 3.9, 1.2, 5.1, 1.6, 4.5, 1.5, 4.5, 1.6, 4.7, 1.5, 4.4, 1.3, 4.1, 1.3, 4.0, 1.3, 4.4, 1.2, 4.6, 1.4, 4.0, 1.2, 3.3, 1.0, 4.2, 1.3, 4.2, 1.2, 4.2, 1.3, 4.3, 1.3, 3.0, 1.1, 4.1, 1.3, 6.0, 2.5, 5.1, 1.9, 5.9, 2.1, 5.6, 1.8, 5.8, 2.2, 6.6, 2.1, 4.5, 1.7, 6.3, 1.8, 5.8, 1.8, 6.1, 2.5, 5.1, 2.0, 5.3, 1.9, 5.5, 2.1, 5.0, 2.0, 5.1, 2.4, 5.3, 2.3, 5.5, 1.8, 6.7, 2.2, 6.9, 2.3, 5.0, 1.5, 5.7, 2.3, 4.9, 2.0, 6.7, 2.0, 4.9, 1.8, 5.7, 2.1, 6.0, 1.8, 4.8, 1.8, 4.9, 1.8, 5.6, 2.1, 5.8, 1.6, 6.1, 1.9, 6.4, 2.0, 5.6, 2.2, 5.1, 1.5, 5.6, 1.4, 6.1, 2.3, 5.6, 2.4, 5.5, 1.8, 4.8, 1.8, 5.4, 2.1, 5.6, 2.4, 5.1, 2.3, 5.1, 1.9, 5.9, 2.3, 5.7, 2.5, 5.2, 2.3, 5.0, 1.9, 5.2, 2.0, 5.4, 2.3, 5.1, 1.8];

        let kmean = KMeans::new(samples, 150, 2);
        let rnd = rand::rngs::StdRng::seed_from_u64(1);
		let conf = KMeansConfig::build().random_generator(rnd).build();
        let res = kmean.kmeans_lloyd(3, 100, KMeans::init_kmeanplusplus, &conf);

        // SHOULD solution
        let should_assignments = vec![1usize, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 0, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
        let should_centroid_distances = vec![0.0059600007, 0.0059600007, 0.028360017, 0.0035599954, 0.0059600007, 0.080360025, 0.00676001, 0.0035599954, 0.0059600007, 0.022759989, 0.0035599954, 0.02116, 0.025159994, 0.15235998, 0.070759974, 0.025160013, 0.049960032, 0.00676001, 0.05956002, 0.0043600043, 0.05876001, 0.025160013, 0.21556, 0.12116004, 0.19395997, 0.02116, 0.042760015, 0.0035599954, 0.0059600007, 0.02116, 0.02116, 0.025160013, 0.022759989, 0.0059600007, 0.0035599954, 0.070759974, 0.028360017, 0.025159994, 0.028360017, 0.0035599954, 0.029160026, 0.029160026, 0.028360017, 0.14436004, 0.21555999, 0.00676001, 0.02116, 0.0059600007, 0.0035599954, 0.0059600007, 0.1676405, 0.06282582, 0.38875192, 0.08912205, 0.114307255, 0.046529524, 0.22393683, 1.1143073, 0.098010965, 0.15578863, 0.7572701, 0.02838137, 0.2146776, 0.1676405, 0.4831962, 0.013196193, 0.06282582, 0.16615912, 0.06282582, 0.22134417, 0.4517149, 0.08912205, 0.38875192, 0.19134419, 0.0035665375, 0.013196193, 0.25912234, 0.5129684, 0.06282582, 0.7572701, 0.3098628, 0.48023304, 0.17949231, 0.47731638, 0.06282582, 0.10097398, 0.18578866, 0.015048049, 0.04060358, 0.08912205, 0.03689988, 0.09615911, 0.11097388, 1.1143073, 0.012085075, 0.033936903, 0.012085075, 0.0035665375, 1.7380108, 0.04060358, 0.34427163, 0.29862052, 0.07775034, 0.06209856, 0.053402513, 0.9512281, 0.15912215, 0.5155767, 0.09166375, 0.4290541, 0.2790552, 0.12818542, 0.01862004, 0.39427257, 0.40079403, 0.16992417, 0.07731599, 1.1764451, 1.6864456, 0.5202333, 0.069054514, 0.5294899, 1.1555758, 0.5631963, 0.008185137, 0.2012288, 0.4517149, 0.5631963, 0.0034026075, 0.23079431, 0.24644595, 0.6012286, 0.023837328, 0.57688177, 0.4203598, 0.2881847, 0.12470677, 0.07731599, 0.4517149, 0.053837437, 0.12470677, 0.34035927, 0.29862052, 0.13861972, 0.20992392, 0.2451419, 0.41383788, 0.18383783, 0.114706814, 0.33818585];
        let should_centroids = vec![4.2925925, 1.3592592, 1.462, 0.24599996, 5.626087, 2.0478265];
		let should_centroid_frequency = vec![54, 50, 46];

        assert_eq!(res.distsum, 31.412888);
        assert_eq!(res.sample_dims, LANES);
        assert_eq!(res.assignments, should_assignments);
        assert_eq!(res.centroid_distances, should_centroid_distances);
        assert_eq!(res.centroids, should_centroids);
		assert_eq!(res.centroid_frequency, should_centroid_frequency);
    }

    #[test]
    fn empty_cluster_handling() {
        let samples = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        let initial_centroids = [2.0, 0.0, 1337.0, 0.0];

        let kmean = KMeans::new(samples, 3, 2);
        let rnd = rand::rngs::StdRng::seed_from_u64(1);
        let conf = KMeansConfig::build().random_generator(rnd).build();

        let res = kmean.kmeans_lloyd(2, 1, |kmean: &KMeans<f64>, state: &mut KMeansState<f64>, _| {
            // p_<array> arrays are padded to p_sample_dims!
            state.centroids[0] = initial_centroids[0];
            state.centroids[1] = initial_centroids[1];
            state.centroids[kmean.p_sample_dims + 0] = initial_centroids[2];
            state.centroids[kmean.p_sample_dims + 1] = initial_centroids[3];
        }, &conf);
        assert_eq!(res.distsum, 0.5);
        assert_eq!(&res.assignments, &[0,0,1]);
        assert_eq!(&res.centroids, &[1.5, 0.0, 3.0, 0.0]);
        assert_eq!(&res.centroid_frequency, &[2,1]);
        assert_eq!(&res.centroid_distances, &[0.25, 0.25, 0.0]);
    }
}
