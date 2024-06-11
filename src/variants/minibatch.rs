use crate::{KMeans, KMeansConfig, KMeansState, memory::*};
use rand::prelude::*;
use rayon::prelude::*;
use std::ops::{Range, DerefMut};
use std::iter::Sum;
use std::ops::{Add, Sub, Mul, Div};
use std::simd::num::SimdFloat;
use std::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};

struct BatchInfo {
	start_idx: usize,
	batch_size: usize
}
impl BatchInfo {
	fn gen_range(&self, stride: usize) -> Range<usize> {
		Range { start: (self.start_idx * stride), end: (self.start_idx * stride + self.batch_size * stride) }
	}
}

pub(crate) struct Minibatch<T, const LANES: usize> where T: Primitive, LaneCount<LANES>: SupportedLaneCount{
	_p: std::marker::PhantomData<T>
}
// TODO
impl<T, const LANES: usize> Minibatch<T, LANES>
where
	T: SimdElement + Copy + Default + Add<Output = T> + Mul<Output = T> + Div<Output = T> + Sub<Output = T> + Sum + Primitive,
	Simd<T, LANES>: Sub<Output = Simd<T, LANES>>
		+ Add<Output = Simd<T, LANES>>
		+ Mul<Output = Simd<T, LANES>>
		+ Div<Output = Simd<T, LANES>>
		+ Sum
		+ SimdFloat<Scalar = T>,
	LaneCount<LANES>: SupportedLaneCount,
{
	fn update_cluster_assignments<'a>(data: &KMeans<T, LANES>, state: &mut KMeansState<T>, batch: &BatchInfo, shuffled_samples: &'a [T], limit_k: Option<usize>) {
		let centroids = &state.centroids;
		let k = limit_k.unwrap_or(state.k);

		// TODO: Switch to par_chunks_exact, when that is merged in rayon (https://github.com/rayon-rs/rayon/pull/629).
		// par_chunks() works, because sample-dimensions are manually padded, so that there is no remainder

		// manually calculate work-packet size, because rayon does not do static scheduling (which is more apropriate here)
        let work_packet_size = batch.batch_size / rayon::current_num_threads();
		shuffled_samples[batch.gen_range(data.p_sample_dims)].par_chunks(data.p_sample_dims)
			.with_min_len(work_packet_size)
			.zip(state.assignments[batch.gen_range(1)].par_iter_mut())
			.zip(state.centroid_distances[batch.gen_range(1)].par_iter_mut())
			.for_each(|((s, assignment), centroid_dist)| {
				let (best_idx, best_dist) = centroids.chunks_exact(data.p_sample_dims).take(k)
					.map(|c| {
						s.chunks_exact(LANES).map(|i| Simd::from_slice(i))
							.zip(c.chunks_exact(LANES).map(|i| Simd::from_slice(i)))
								.map(|(sp,cp)| sp - cp)         // <sample> - <centroid>
								.map(|v| v * v)                 // <vec_components> ^2
								.sum::<Simd::<T,LANES>>()     // sum(<vec_components>^2)
								.reduce_sum()                   // sum(sum(<vec_components>^2))
					}).enumerate()
					.min_by(|(_,d0), (_,d1)| d0.partial_cmp(d1).unwrap()).unwrap();
				*assignment = best_idx;
				*centroid_dist = best_dist;
			});
	}

	fn update_centroids<'a>(data: &KMeans<T, LANES>, state: &mut KMeansState<T>, batch: &BatchInfo, shuffled_samples: &'a [T]) {
		let centroid_frequency = &mut state.centroid_frequency;
		let centroids = &mut state.centroids;
		let assignments = &state.assignments;

		shuffled_samples[batch.gen_range(data.p_sample_dims)].chunks_exact(data.p_sample_dims)
			.zip(assignments[batch.gen_range(1)].iter().cloned())
			.for_each(|(sample, assignment)| {
				centroid_frequency[assignment] += 1;
				let learn_rate = T::one() / T::from(centroid_frequency[assignment]).unwrap();
				let inv_learn_rate = T::one() - learn_rate;
				centroids.iter_mut().skip(assignment * data.p_sample_dims).take(data.p_sample_dims)
					.zip(sample.iter().cloned())
					.for_each(|(c, s)| {
						*c = inv_learn_rate * *c + learn_rate * s;
					});
			});
	}

	fn shuffle_samples<'a>(data: &KMeans<T, LANES>, config: &KMeansConfig<'a, T>) -> (Vec<usize>, Vec<T>) {
		let mut idxs: Vec<usize> = (0..data.sample_cnt).collect();
		idxs.shuffle(config.rnd.borrow_mut().deref_mut());

		let mut shuffled_samples = AlignedFloatVec::<LANES>::new_uninitialized(data.p_samples.len());
		shuffled_samples.chunks_exact_mut(data.p_sample_dims)
			.zip(idxs.iter().map(|i| &data.p_samples[(i * data.p_sample_dims)..(i * data.p_sample_dims) + data.p_sample_dims] ))
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

	#[inline(always)] pub fn calculate<'a, F>(data: &KMeans<T, LANES>, batch_size: usize, k: usize, max_iter: usize, init: F, config: &KMeansConfig<'a, T>) -> KMeansState<T>
				where for<'c> F: FnOnce(&KMeans<T, LANES>, &mut KMeansState<T>, &KMeansConfig<'c, T>) {
		assert!(k <= data.sample_cnt);
		assert!(batch_size <= data.sample_cnt);

		// Copy and shuffle sample_data, then only take consecutive blocks (with batch_size) from there
		let (shuffle_idxs, shuffled_samples) = Self::shuffle_samples(data, config);


		let mut state = KMeansState::new::<LANES>(data.sample_cnt, data.p_sample_dims, k);
        state.distsum = T::infinity();

		// Initialize clusters and notify subscriber
		init(&data, &mut state, config);
        (config.init_done)(&state);
		let mut abort_strategy = config.abort_strategy.create_logic();

		// Update cluster assignments for all samples, to get rid of the INFINITES in centroid_distances
		Self::update_cluster_assignments(data, &mut state, &BatchInfo{start_idx: 0, batch_size: data.sample_cnt}, &shuffled_samples, None);

		for i in 1..=max_iter {
			// Only shuffle a beginning index for a consecutive block within the shuffled samples as batch
			let batch = BatchInfo {
				batch_size,
				start_idx: config.rnd.borrow_mut().gen_range(0..data.sample_cnt - batch_size)
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

		let (assignments, centroid_frequency, centroid_distances, distsum) =
			(&state.assignments, &mut state.centroid_frequency, &mut state.centroid_distances, &mut state.distsum);
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
	use crate::AbortStrategy;

	#[test]
    fn iris_dataset_f64() where {
        let samples = vec![1.4f64, 0.2, 1.4, 0.2, 1.3, 0.2, 1.5, 0.2, 1.4, 0.2, 1.7, 0.4, 1.4, 0.3, 1.5, 0.2, 1.4, 0.2, 1.5, 0.1, 1.5, 0.2, 1.6, 0.2, 1.4, 0.1, 1.1, 0.1, 1.2, 0.2, 1.5, 0.4, 1.3, 0.4, 1.4, 0.3, 1.7, 0.3, 1.5, 0.3, 1.7, 0.2, 1.5, 0.4, 1.0, 0.2, 1.7, 0.5, 1.9, 0.2, 1.6, 0.2, 1.6, 0.4, 1.5, 0.2, 1.4, 0.2, 1.6, 0.2, 1.6, 0.2, 1.5, 0.4, 1.5, 0.1, 1.4, 0.2, 1.5, 0.2, 1.2, 0.2, 1.3, 0.2, 1.4, 0.1, 1.3, 0.2, 1.5, 0.2, 1.3, 0.3, 1.3, 0.3, 1.3, 0.2, 1.6, 0.6, 1.9, 0.4, 1.4, 0.3, 1.6, 0.2, 1.4, 0.2, 1.5, 0.2, 1.4, 0.2, 4.7, 1.4, 4.5, 1.5, 4.9, 1.5, 4.0, 1.3, 4.6, 1.5, 4.5, 1.3, 4.7, 1.6, 3.3, 1.0, 4.6, 1.3, 3.9, 1.4, 3.5, 1.0, 4.2, 1.5, 4.0, 1.0, 4.7, 1.4, 3.6, 1.3, 4.4, 1.4, 4.5, 1.5, 4.1, 1.0, 4.5, 1.5, 3.9, 1.1, 4.8, 1.8, 4.0, 1.3, 4.9, 1.5, 4.7, 1.2, 4.3, 1.3, 4.4, 1.4, 4.8, 1.4, 5.0, 1.7, 4.5, 1.5, 3.5, 1.0, 3.8, 1.1, 3.7, 1.0, 3.9, 1.2, 5.1, 1.6, 4.5, 1.5, 4.5, 1.6, 4.7, 1.5, 4.4, 1.3, 4.1, 1.3, 4.0, 1.3, 4.4, 1.2, 4.6, 1.4, 4.0, 1.2, 3.3, 1.0, 4.2, 1.3, 4.2, 1.2, 4.2, 1.3, 4.3, 1.3, 3.0, 1.1, 4.1, 1.3, 6.0, 2.5, 5.1, 1.9, 5.9, 2.1, 5.6, 1.8, 5.8, 2.2, 6.6, 2.1, 4.5, 1.7, 6.3, 1.8, 5.8, 1.8, 6.1, 2.5, 5.1, 2.0, 5.3, 1.9, 5.5, 2.1, 5.0, 2.0, 5.1, 2.4, 5.3, 2.3, 5.5, 1.8, 6.7, 2.2, 6.9, 2.3, 5.0, 1.5, 5.7, 2.3, 4.9, 2.0, 6.7, 2.0, 4.9, 1.8, 5.7, 2.1, 6.0, 1.8, 4.8, 1.8, 4.9, 1.8, 5.6, 2.1, 5.8, 1.6, 6.1, 1.9, 6.4, 2.0, 5.6, 2.2, 5.1, 1.5, 5.6, 1.4, 6.1, 2.3, 5.6, 2.4, 5.5, 1.8, 4.8, 1.8, 5.4, 2.1, 5.6, 2.4, 5.1, 2.3, 5.1, 1.9, 5.9, 2.3, 5.7, 2.5, 5.2, 2.3, 5.0, 1.9, 5.2, 2.0, 5.4, 2.3, 5.1, 1.8];

        let kmean: KMeans<f64, 8> = KMeans::new(samples, 150, 2);
        let rnd = rand::rngs::StdRng::seed_from_u64(1);
		let conf = KMeansConfig::build()
			.random_generator(rnd)
			.abort_strategy(AbortStrategy::NoImprovementForXIterations {
				x: 5, threshold: 0.0005f64, abort_on_negative: true
			})
			.build();
        let res = kmean.kmeans_minibatch(30, 3, 100, KMeans::init_kmeanplusplus, &conf);

        // SHOULD solution
        let should_assignments = vec![2usize, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1];
        let should_centroid_distances = vec![0.006270216341104825, 0.006270216341104825, 0.03003833228313378, 0.002502100399075829, 0.006270216341104825, 0.07931369460197435, 0.008444129384583074, 0.002502100399075829, 0.006270216341104825, 0.02032818735559758, 0.002502100399075829, 0.018733984457046868, 0.024096303297626576, 0.15540065112371346, 0.07380644822516282, 0.026849926486032336, 0.05438615837009029, 0.008444129384583074, 0.05713978155849608, 0.004676013442554076, 0.054965868515017835, 0.026849926486032336, 0.22134268010922073, 0.12148760764545258, 0.1874296366309598, 0.018733984457046868, 0.043081810544003375, 0.002502100399075829, 0.006270216341104825, 0.018733984457046868, 0.018733984457046868, 0.026849926486032336, 0.02032818735559758, 0.006270216341104825, 0.002502100399075829, 0.07380644822516282, 0.03003833228313378, 0.024096303297626576, 0.03003833228313378, 0.002502100399075829, 0.03221224532661203, 0.03221224532661203, 0.03003833228313378, 0.14742963663095984, 0.2117774627179163, 0.008444129384583074, 0.018733984457046868, 0.006270216341104825, 0.002502100399075829, 0.006270216341104825, 0.10015615699730515, 0.020215328594937368, 0.27299639368369644, 0.16133958894996572, 0.05341059486712688, 0.027315920310914045, 0.1330555652813285, 1.3496236126186034, 0.06051118658310356, 0.23459402681978783, 0.9560141451629824, 0.04062952977836832, 0.32199047652393076, 0.10015615699730515, 0.628558523861207, 0.000570358180736065, 0.020215328594937368, 0.2551857427961206, 0.020215328594937368, 0.3352449143937528, 0.3191502398375412, 0.16133958894996572, 0.27299639368369644, 0.14725674871328187, 0.020925387766534752, 0.000570358180736065, 0.17335142326949454, 0.459091068239909, 0.020215328594937368, 0.9560141451629824, 0.4420496481215632, 0.6424046777073615, 0.28169461853576455, 0.5458366303700866, 0.020215328594937368, 0.04666503273694906, 0.1066058611393168, 0.014120654038724395, 0.0945348552221556, 0.16133958894996572, 0.04767094989671277, 0.04696089072511523, 0.1948898848079541, 1.3496236126186034, 0.047730121494344996, 0.08128041735233338, 0.047730121494344996, 0.020925387766534752, 2.0164875179440456, 0.0945348552221556, 0.1918208160388422, 0.5558031169238039, 0.0067765682512331785, 0.13845798418043886, 0.009962408959197902, 0.6119978071892828, 0.09311473687896066, 0.32367922311848873, 0.09137833816273876, 0.23828099302999206, 0.5254491346229186, 0.30872347090610364, 0.10093586028663301, 0.678988957631768, 0.6040332054193777, 0.3073075417025626, 0.19199780718928866, 0.7881040018795485, 1.2106703735609636, 0.3861916599558857, 0.05314824966716249, 0.6052449143937547, 0.7888119664813189, 0.4123455061097314, 0.01385621426893302, 0.1242986921450388, 0.3191502398375412, 0.4123455061097314, 0.04739603727778319, 0.25208630276450916, 0.12040488683530348, 0.3494314355078689, 0.057042054976897995, 0.5193869262280748, 0.5398739133839799, 0.11898895763176244, 0.13633409037512745, 0.19199780718928866, 0.3191502398375412, 0.1744756832954827, 0.13633409037512745, 0.5543871877202629, 0.5558031169238039, 0.04606860364946264, 0.17244028506539208, 0.42084736471141215, 0.6119904765239322, 0.39190931161406795, 0.21376771869371214, 0.6061570992246891];
        let should_centroids = vec![4.384023668639052, 1.4177514792899417, 5.81769911504425, 2.101769911504426, 1.468840579710145, 0.23913043478260876];
		let should_centroid_frequency = vec![59, 41, 50];

        assert_eq!(res.distsum, 32.582058324288155);
        assert_eq!(res.sample_dims, 8);
        assert_eq!(res.assignments, should_assignments);
        assert_eq!(res.centroid_distances, should_centroid_distances);
        assert_eq!(res.centroids, should_centroids);
		assert_eq!(res.centroid_frequency, should_centroid_frequency);
    }

    #[test]
    fn iris_dataset_f32() where {
        let samples = vec![1.4f32, 0.2, 1.4, 0.2, 1.3, 0.2, 1.5, 0.2, 1.4, 0.2, 1.7, 0.4, 1.4, 0.3, 1.5, 0.2, 1.4, 0.2, 1.5, 0.1, 1.5, 0.2, 1.6, 0.2, 1.4, 0.1, 1.1, 0.1, 1.2, 0.2, 1.5, 0.4, 1.3, 0.4, 1.4, 0.3, 1.7, 0.3, 1.5, 0.3, 1.7, 0.2, 1.5, 0.4, 1.0, 0.2, 1.7, 0.5, 1.9, 0.2, 1.6, 0.2, 1.6, 0.4, 1.5, 0.2, 1.4, 0.2, 1.6, 0.2, 1.6, 0.2, 1.5, 0.4, 1.5, 0.1, 1.4, 0.2, 1.5, 0.2, 1.2, 0.2, 1.3, 0.2, 1.4, 0.1, 1.3, 0.2, 1.5, 0.2, 1.3, 0.3, 1.3, 0.3, 1.3, 0.2, 1.6, 0.6, 1.9, 0.4, 1.4, 0.3, 1.6, 0.2, 1.4, 0.2, 1.5, 0.2, 1.4, 0.2, 4.7, 1.4, 4.5, 1.5, 4.9, 1.5, 4.0, 1.3, 4.6, 1.5, 4.5, 1.3, 4.7, 1.6, 3.3, 1.0, 4.6, 1.3, 3.9, 1.4, 3.5, 1.0, 4.2, 1.5, 4.0, 1.0, 4.7, 1.4, 3.6, 1.3, 4.4, 1.4, 4.5, 1.5, 4.1, 1.0, 4.5, 1.5, 3.9, 1.1, 4.8, 1.8, 4.0, 1.3, 4.9, 1.5, 4.7, 1.2, 4.3, 1.3, 4.4, 1.4, 4.8, 1.4, 5.0, 1.7, 4.5, 1.5, 3.5, 1.0, 3.8, 1.1, 3.7, 1.0, 3.9, 1.2, 5.1, 1.6, 4.5, 1.5, 4.5, 1.6, 4.7, 1.5, 4.4, 1.3, 4.1, 1.3, 4.0, 1.3, 4.4, 1.2, 4.6, 1.4, 4.0, 1.2, 3.3, 1.0, 4.2, 1.3, 4.2, 1.2, 4.2, 1.3, 4.3, 1.3, 3.0, 1.1, 4.1, 1.3, 6.0, 2.5, 5.1, 1.9, 5.9, 2.1, 5.6, 1.8, 5.8, 2.2, 6.6, 2.1, 4.5, 1.7, 6.3, 1.8, 5.8, 1.8, 6.1, 2.5, 5.1, 2.0, 5.3, 1.9, 5.5, 2.1, 5.0, 2.0, 5.1, 2.4, 5.3, 2.3, 5.5, 1.8, 6.7, 2.2, 6.9, 2.3, 5.0, 1.5, 5.7, 2.3, 4.9, 2.0, 6.7, 2.0, 4.9, 1.8, 5.7, 2.1, 6.0, 1.8, 4.8, 1.8, 4.9, 1.8, 5.6, 2.1, 5.8, 1.6, 6.1, 1.9, 6.4, 2.0, 5.6, 2.2, 5.1, 1.5, 5.6, 1.4, 6.1, 2.3, 5.6, 2.4, 5.5, 1.8, 4.8, 1.8, 5.4, 2.1, 5.6, 2.4, 5.1, 2.3, 5.1, 1.9, 5.9, 2.3, 5.7, 2.5, 5.2, 2.3, 5.0, 1.9, 5.2, 2.0, 5.4, 2.3, 5.1, 1.8];

        let kmean: KMeans<f32, 8> = KMeans::new(samples, 150, 2);
        let rnd = rand::rngs::StdRng::seed_from_u64(1);
		let conf = KMeansConfig::build()
			.random_generator(rnd)
			.abort_strategy(AbortStrategy::NoImprovementForXIterations {
				x: 5, threshold: 0.0005f32, abort_on_negative: true
			})
			.build();
        let res = kmean.kmeans_minibatch(30, 3, 100, KMeans::init_kmeanplusplus, &conf);

        // SHOULD solution
        let should_assignments = vec![1usize, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 0, 2, 2, 0, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
        let should_centroid_distances = vec![0.006600124, 0.006600124, 0.03077228, 0.0024279766, 0.006600124, 0.07818967, 0.008653102, 0.0024279766, 0.006600124, 0.020374997, 0.0024279766, 0.01825584, 0.024547143, 0.15706356, 0.074944384, 0.026533935, 0.05487824, 0.008653102, 0.05613669, 0.0044809557, 0.05408371, 0.026533935, 0.2232887, 0.12024263, 0.18573938, 0.01825584, 0.042361796, 0.0024279766, 0.006600124, 0.01825584, 0.01825584, 0.026533935, 0.020374997, 0.006600124, 0.0024279766, 0.074944384, 0.03077228, 0.024547143, 0.03077228, 0.0024279766, 0.03282526, 0.03282526, 0.03077228, 0.14646776, 0.20984533, 0.008653102, 0.01825584, 0.006600124, 0.0024279766, 0.006600124, 0.17038915, 0.06441513, 0.39296725, 0.08712144, 0.11655308, 0.047811467, 0.22699283, 1.10725, 0.09994941, 0.15328519, 0.7515259, 0.028001154, 0.21221593, 0.17038915, 0.47856957, 0.013975311, 0.06441513, 0.16435398, 0.06441513, 0.21837968, 0.45573482, 0.08712144, 0.39296725, 0.19378546, 0.0035354628, 0.013975311, 0.26252753, 0.6217088, 0.06441513, 0.7515259, 0.3062418, 0.4758019, 0.17668152, 0.5804291, 0.06441513, 0.10271698, 0.18869099, 0.015673485, 0.039259486, 0.08712144, 0.037371628, 0.09825124, 0.10881959, 1.10725, 0.011397488, 0.033095635, 0.011397488, 0.0035354628, 1.7291378, 0.039259486, 0.2808543, 0.3965715, 0.03799805, 0.077856295, 0.02871239, 0.79399645, 0.16101883, 0.4138551, 0.07385591, 0.34885404, 0.37528563, 0.19257082, 0.045998804, 0.50728565, 0.49014217, 0.22742727, 0.10985647, 1.0007101, 1.4754245, 0.5251051, 0.059426643, 0.6592857, 0.9832819, 0.56787276, 0.0019984138, 0.14985548, 0.45573482, 0.56787276, 0.0139986295, 0.21642761, 0.17656931, 0.47928306, 0.032712772, 0.677243, 0.4429998, 0.21142577, 0.13014106, 0.10985647, 0.45573482, 0.09799895, 0.13014106, 0.43142796, 0.3965715, 0.09542628, 0.1968549, 0.31942782, 0.52857155, 0.2632855, 0.15542717, 0.4378574];
        let should_centroids = vec![4.28931, 1.3584908, 1.4708607, 0.23973511, 5.710001, 2.0564294];
		let should_centroid_frequency = vec![56, 50, 44];

        assert_eq!(res.distsum, 31.732794);
        assert_eq!(res.sample_dims, 8);
        assert_eq!(res.assignments, should_assignments);
        assert_eq!(res.centroid_distances, should_centroid_distances);
        assert_eq!(res.centroids, should_centroids);
		assert_eq!(res.centroid_frequency, should_centroid_frequency);
    }
}