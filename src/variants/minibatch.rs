use crate::{KMeans, KMeansEvt, KMeansState, memory::*};
use packed_simd::{Simd, SimdArray};
use rand::prelude::*;
use rayon::prelude::*;
use std::ops::Range;

struct BatchInfo {
	start_idx: usize,
	batch_size: usize
}
impl BatchInfo {
	fn gen_range(&self, stride: usize) -> Range<usize> {
		Range { start: (self.start_idx * stride), end: (self.start_idx * stride + self.batch_size * stride) }
	}
}

pub(crate) struct Minibatch<T> where T: Primitive, [T;LANES]: SimdArray, Simd<[T;LANES]>: SimdWrapper<T>{
	_p: std::marker::PhantomData<T>
}
impl<T> Minibatch<T> where T: Primitive, [T;LANES]: SimdArray, Simd<[T;LANES]>: SimdWrapper<T> {
	fn update_cluster_assignments<'a>(data: &KMeans<T>, state: &mut KMeansState<T>, batch: &BatchInfo, shuffled_samples: &'a [T], limit_k: Option<usize>) {
		let centroids = &state.centroids;
		let k = limit_k.unwrap_or(state.k);

		shuffled_samples[batch.gen_range(data.p_sample_dims)].par_chunks_exact(data.p_sample_dims)
			.zip(state.assignments[batch.gen_range(1)].par_iter_mut())
			.zip(state.centroid_distances[batch.gen_range(1)].par_iter_mut())
			.for_each(|((s, assignment), centroid_dist)| {
				let (best_idx, best_dist) = centroids.chunks_exact(data.p_sample_dims).take(k)
					.map(|c| {
						s.chunks_exact(LANES).map(|i| unsafe { Simd::<[T;LANES]>::from_slice_aligned_unchecked(i) })
							.zip(c.chunks_exact(LANES).map(|i| unsafe { Simd::<[T;LANES]>::from_slice_aligned_unchecked(i) }))
								.map(|(sp,cp)| sp - cp)         // <sample> - <centroid>
								.map(|v| v * v)                 // <vec_components> ^2
								.sum::<Simd::<[T;LANES]>>()     // sum(<vec_components>^2)
								.sum()
					}).enumerate()
					.min_by(|(_,d0), (_,d1)| d0.partial_cmp(d1).unwrap()).unwrap();
				*assignment = best_idx;
				*centroid_dist = best_dist;
			});
	}

	fn update_centroids<'a>(data: &KMeans<T>, state: &mut KMeansState<T>, batch: &BatchInfo, shuffled_samples: &'a [T]) {
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

	fn shuffle_samples<'a>(data: &KMeans<T>, rnd: &'a mut dyn RngCore) -> (Vec<usize>, Vec<T>) {
		let mut idxs: Vec<usize> = (0..data.sample_cnt).collect();
		idxs.shuffle(rnd);

		let mut shuffled_samples = AlignedFloatVec::new_uninitialized(data.p_samples.len());
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

	#[inline(always)] pub fn calculate<'a, 'b, F>(data: &KMeans<T>, batch_size: usize, k: usize, max_iter: usize, init: F, rnd: &'a mut dyn RngCore, evt: KMeansEvt<'b, T>) -> KMeansState<'b, T>
				where for<'c> F: FnOnce(&KMeans<T>, &mut KMeansState<T>, &'c mut dyn RngCore) {
		assert!(k <= data.sample_cnt);
		assert!(batch_size <= data.sample_cnt);

		// Copy and shuffle sample_data, then only take consecutive blocks (with batch_size) from there
		let (shuffle_idxs, shuffled_samples) = Self::shuffle_samples(data, rnd);


		let mut state = KMeansState::new(data.sample_cnt, data.p_sample_dims, k, evt);
        state.distsum = T::infinity();
		// Count how many times the distsum did not improve, exit after 5 iterations without improvement
		let mut improvement_counter = 0;

		// Initialize clusters and notify subscriber
		init(&data, &mut state, rnd);
        (state.evt.init_done)(&state);
		// Update cluster assignments for all samples, to get rid of the INFINITES in centroid_distances
		Self::update_cluster_assignments(data, &mut state, &BatchInfo{start_idx: 0, batch_size: data.sample_cnt}, &shuffled_samples, None);

		for i in 1..=max_iter {
			// Only shuffle a beginning index for a consecutive block within the shuffled samples as batch
			let batch = BatchInfo {
				batch_size,
				start_idx: rnd.gen_range(0, data.sample_cnt - batch_size)
			};

			Self::update_cluster_assignments(data, &mut state, &batch, &shuffled_samples, None);
			let new_distsum = state.centroid_distances.iter().cloned().sum();
			Self::update_centroids(data, &mut state, &batch, &shuffled_samples);

			// Notify subscriber about finished iteration
			(state.evt.iteration_done)(&state, i, new_distsum);

			let improvement = state.distsum - new_distsum;
            if improvement < T::from(0.0005).unwrap() {
				improvement_counter += 1;
				// If there was no improvement over a course of 5 iterations, abort.
				// But directly abort, if there was a negative "improvement".
				if improvement < T::zero() || improvement_counter == 5 {
					break;
				}
            } else {
				improvement_counter = 0;
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

	#[test]
    fn iris_dataset_f64() where {
        let samples = vec![1.4f64, 0.2, 1.4, 0.2, 1.3, 0.2, 1.5, 0.2, 1.4, 0.2, 1.7, 0.4, 1.4, 0.3, 1.5, 0.2, 1.4, 0.2, 1.5, 0.1, 1.5, 0.2, 1.6, 0.2, 1.4, 0.1, 1.1, 0.1, 1.2, 0.2, 1.5, 0.4, 1.3, 0.4, 1.4, 0.3, 1.7, 0.3, 1.5, 0.3, 1.7, 0.2, 1.5, 0.4, 1.0, 0.2, 1.7, 0.5, 1.9, 0.2, 1.6, 0.2, 1.6, 0.4, 1.5, 0.2, 1.4, 0.2, 1.6, 0.2, 1.6, 0.2, 1.5, 0.4, 1.5, 0.1, 1.4, 0.2, 1.5, 0.2, 1.2, 0.2, 1.3, 0.2, 1.4, 0.1, 1.3, 0.2, 1.5, 0.2, 1.3, 0.3, 1.3, 0.3, 1.3, 0.2, 1.6, 0.6, 1.9, 0.4, 1.4, 0.3, 1.6, 0.2, 1.4, 0.2, 1.5, 0.2, 1.4, 0.2, 4.7, 1.4, 4.5, 1.5, 4.9, 1.5, 4.0, 1.3, 4.6, 1.5, 4.5, 1.3, 4.7, 1.6, 3.3, 1.0, 4.6, 1.3, 3.9, 1.4, 3.5, 1.0, 4.2, 1.5, 4.0, 1.0, 4.7, 1.4, 3.6, 1.3, 4.4, 1.4, 4.5, 1.5, 4.1, 1.0, 4.5, 1.5, 3.9, 1.1, 4.8, 1.8, 4.0, 1.3, 4.9, 1.5, 4.7, 1.2, 4.3, 1.3, 4.4, 1.4, 4.8, 1.4, 5.0, 1.7, 4.5, 1.5, 3.5, 1.0, 3.8, 1.1, 3.7, 1.0, 3.9, 1.2, 5.1, 1.6, 4.5, 1.5, 4.5, 1.6, 4.7, 1.5, 4.4, 1.3, 4.1, 1.3, 4.0, 1.3, 4.4, 1.2, 4.6, 1.4, 4.0, 1.2, 3.3, 1.0, 4.2, 1.3, 4.2, 1.2, 4.2, 1.3, 4.3, 1.3, 3.0, 1.1, 4.1, 1.3, 6.0, 2.5, 5.1, 1.9, 5.9, 2.1, 5.6, 1.8, 5.8, 2.2, 6.6, 2.1, 4.5, 1.7, 6.3, 1.8, 5.8, 1.8, 6.1, 2.5, 5.1, 2.0, 5.3, 1.9, 5.5, 2.1, 5.0, 2.0, 5.1, 2.4, 5.3, 2.3, 5.5, 1.8, 6.7, 2.2, 6.9, 2.3, 5.0, 1.5, 5.7, 2.3, 4.9, 2.0, 6.7, 2.0, 4.9, 1.8, 5.7, 2.1, 6.0, 1.8, 4.8, 1.8, 4.9, 1.8, 5.6, 2.1, 5.8, 1.6, 6.1, 1.9, 6.4, 2.0, 5.6, 2.2, 5.1, 1.5, 5.6, 1.4, 6.1, 2.3, 5.6, 2.4, 5.5, 1.8, 4.8, 1.8, 5.4, 2.1, 5.6, 2.4, 5.1, 2.3, 5.1, 1.9, 5.9, 2.3, 5.7, 2.5, 5.2, 2.3, 5.0, 1.9, 5.2, 2.0, 5.4, 2.3, 5.1, 1.8];

        let kmean = KMeans::new(samples, 150, 2);
        let mut rnd = rand::rngs::StdRng::seed_from_u64(1);
        let res = kmean.kmeans_minibatch(30, 3, 100, KMeans::init_kmeanplusplus, &mut rnd, None);

        // SHOULD solution
        let should_assignments = vec![2usize, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let should_centroid_distances = vec![0.007848555588905955, 0.007848555588905955, 0.032781614291274165, 0.0029154968865377088, 0.007848555588905955, 0.07397625898746553, 0.008311995341738106, 0.0029154968865377088, 0.007848555588905955, 0.022452057133705558, 0.0029154968865377088, 0.017982438184169496, 0.027385115836073805, 0.16218429194317843, 0.07771467299364243, 0.02384237639220202, 0.05370849379693847, 0.008311995341738106, 0.05351281923463337, 0.00337893663936986, 0.053049379481801225, 0.02384237639220202, 0.22758079039837886, 0.11443969874029768, 0.1831832620770647, 0.017982438184169496, 0.03890931768983381, 0.0029154968865377088, 0.007848555588905955, 0.017982438184169496, 0.017982438184169496, 0.02384237639220202, 0.022452057133705558, 0.007848555588905955, 0.0029154968865377088, 0.07771467299364243, 0.032781614291274165, 0.027385115836073805, 0.032781614291274165, 0.0029154968865377088, 0.03324505404410631, 0.03324505404410631, 0.032781614291274165, 0.13983619719549809, 0.20411014158272903, 0.008311995341738106, 0.017982438184169496, 0.007848555588905955, 0.0029154968865377088, 0.007848555588905955, 0.08192770852569796, 0.01443648045552487, 0.24306805940289053, 0.18499788396429886, 0.04159437519236606, 0.02078735764850551, 0.11557683133271734, 1.414418936595881, 0.04794525238534671, 0.2646645506309673, 1.0087347260695632, 0.05296279624500081, 0.34452419975376986, 0.08192770852569796, 0.6763663050169334, 0.0004540243151738595, 0.01443648045552487, 0.2716820944906114, 0.01443648045552487, 0.3641908664204382, 0.29638384887657776, 0.18499788396429886, 0.24306805940289053, 0.12827858571867862, 0.026471568174822893, 0.0004540243151738595, 0.14908560326253906, 0.42387507694675086, 0.01443648045552487, 1.0087347260695632, 0.47703297168359704, 0.6830505155432456, 0.311015427823948, 0.504208410280082, 0.01443648045552487, 0.04126104185903458, 0.08875226992920762, 0.013629462911664171, 0.11215577870114042, 0.18499788396429886, 0.04680490150815453, 0.034769813788856394, 0.21817332256078922, 1.414418936595881, 0.059313673437981454, 0.0924891120344718, 0.059313673437981454, 0.026471568174822893, 2.0997698137888663, 0.11215577870114042, 0.25483251742013485, 0.402740278949952, 0.03792588074971878, 0.09522621820753883, 0.021030492673229546, 0.8002093453166398, 0.0880856032625442, 0.43750968277446023, 0.09302149379808777, 0.3237301552154091, 0.3747425286687374, 0.20053555454050082, 0.04233532956862054, 0.5058448908734625, 0.46275152754387905, 0.2085445534156424, 0.12632858041226425, 1.0011092328307007, 1.4709067581400357, 0.3502259541397315, 0.04413510459674036, 0.5771908664204388, 0.9971047333931298, 0.3835417436134196, 0.0001306051591695878, 0.17081676938863688, 0.29638384887657776, 0.3835417436134196, 0.011232967363895138, 0.24901699436051686, 0.1917166569026966, 0.4904118200073063, 0.0232352170826806, 0.4773838488765723, 0.48721721933239726, 0.19972565577783818, 0.10723971652025135, 0.12632858041226425, 0.29638384887657776, 0.0934376917733458, 0.10723971652025135, 0.41074927782509363, 0.402740278949952, 0.08193038018728956, 0.16813960403431127, 0.29964691562036755, 0.5338426411546771, 0.26364016646401134, 0.13744219121091655, 0.45073802923116657];
        let should_centroids = vec![4.414210526315793, 1.4158771929824516, 5.705511811023627, 2.089988751406073, 1.4746652935118412, 0.24768280123583925];
		let should_centroid_frequency = vec![58, 42, 50];

        assert_eq!(res.distsum, 32.11922329828416);
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
        let mut rnd = rand::rngs::StdRng::seed_from_u64(1);
        let res = kmean.kmeans_minibatch(30, 3, 100, KMeans::init_kmeanplusplus, &mut rnd, None);

        // SHOULD solution
        let should_assignments = vec![1usize, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 0, 2, 2, 0, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
        let should_centroid_distances = vec![0.007960148, 0.007960148, 0.032975376, 0.0029449286, 0.007960148, 0.0736325, 0.0083191395, 0.0029449286, 0.007960148, 0.022585936, 0.0029449286, 0.017929718, 0.027601156, 0.16264677, 0.07799055, 0.02366291, 0.053693358, 0.0083191395, 0.05327351, 0.00330392, 0.05291452, 0.02366291, 0.22802101, 0.11399148, 0.18288405, 0.017929718, 0.0386477, 0.0029449286, 0.007960148, 0.017929718, 0.017929718, 0.02366291, 0.022585936, 0.007960148, 0.0029449286, 0.07799055, 0.032975376, 0.027601156, 0.032975376, 0.0029449286, 0.033334367, 0.033334367, 0.032975376, 0.13936569, 0.20360203, 0.0083191395, 0.017929718, 0.007960148, 0.0029449286, 0.007960148, 0.10170213, 0.025305545, 0.2803908, 0.15415739, 0.05907679, 0.023013817, 0.14399388, 1.3243209, 0.056785062, 0.23153187, 0.9318633, 0.043991756, 0.30071977, 0.10170213, 0.61907244, 0.00038839362, 0.025305545, 0.23449111, 0.025305545, 0.31809425, 0.34005713, 0.15415739, 0.2803908, 0.13941038, 0.015471214, 0.00038839362, 0.17547369, 0.47645372, 0.025305545, 0.9318633, 0.4243231, 0.61940587, 0.2692401, 0.54202217, 0.025305545, 0.056451425, 0.112848, 0.009242535, 0.08792873, 0.15415739, 0.038096637, 0.04793092, 0.18301149, 1.3243209, 0.041700028, 0.07055413, 0.041700028, 0.015471214, 1.9941528, 0.08792873, 0.29539934, 0.3488183, 0.056005783, 0.07821336, 0.034605745, 0.8749969, 0.107597314, 0.47720495, 0.09221093, 0.37239802, 0.324417, 0.16281559, 0.028010666, 0.44741812, 0.4268119, 0.18521039, 0.10121457, 1.0875942, 1.5771911, 0.39416197, 0.053205587, 0.59041923, 1.0763967, 0.43382835, 0.002008188, 0.18620843, 0.34005713, 0.43382835, 0.0050094463, 0.24101347, 0.21880579, 0.54540104, 0.02060817, 0.5279331, 0.45581853, 0.2412006, 0.11180563, 0.10121457, 0.34005713, 0.07101184, 0.11180563, 0.3712131, 0.3488183, 0.107203186, 0.18440303, 0.26821196, 0.47181943, 0.22141585, 0.12220924, 0.39321962];
        let should_centroids = vec![4.3811436, 1.3942707, 1.4750761, 0.24820505, 5.665006, 2.0720065];
		let should_centroid_frequency = vec![56, 50, 44];

        assert_eq!(res.distsum, 31.785213);
        assert_eq!(res.sample_dims, LANES);
        assert_eq!(res.assignments, should_assignments);
        assert_eq!(res.centroid_distances, should_centroid_distances);
        assert_eq!(res.centroids, should_centroids);
		assert_eq!(res.centroid_frequency, should_centroid_frequency);
    }
}