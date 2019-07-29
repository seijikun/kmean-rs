#![feature(test)]
extern crate test;
use rayon::prelude::*;
use rand::{
    prelude::*,
    distributions::{
        weighted::WeightedIndex,
        uniform::SampleUniform
    }
};
use num::{NumCast, Zero, Float};
use std::{
    ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign},
    fmt::{Display, Debug, LowerExp},
    iter::Sum
};
use packed_simd::{Simd, SimdArray};


// TODO: Remove this and use const_generics, as soon as they are stable and the compiler stops crashing :)
const LANES: usize = 8;




pub trait Primitive: Add + AddAssign + Sum + Sub + SubAssign + Zero + Float + NumCast + SampleUniform
                + PartialOrd + Copy + Default + Display + Debug + Sync + Send + LowerExp
                + for<'a> AddAssign<&'a Self> + for<'a> Sub<&'a Self> {}
impl Primitive for f32 {}
impl Primitive for f64 {}

pub trait SimdWrapper<T> : Sized + Add<Output = Self> + AddAssign + Sub<Output = Self> + SubAssign
                    + Mul<Output = Self> + MulAssign + Div<Output = Self> + DivAssign + Sum where [T;LANES]: SimdArray {
    unsafe fn from_slice_aligned_unchecked(src: &[T]) -> Self;
    unsafe fn write_to_slice_aligned_unchecked(self, slice: &mut [T]);
    fn splat(single: T) -> Self;
    fn sum(self) -> T;
}
macro_rules! impl_simd_wrapper {
    ($simd:ty, $primitive:ty, $lanes:expr) => {
        impl SimdWrapper<$primitive> for $simd {
            unsafe fn from_slice_aligned_unchecked(src: &[$primitive]) -> Self { <$simd>::from_slice_aligned_unchecked(src) }
            unsafe fn write_to_slice_aligned_unchecked(self, slice: &mut [$primitive]) { self.write_to_slice_aligned_unchecked(slice); }
            fn splat(single: $primitive) -> Self { <$simd>::splat(single) }
            fn sum(self) -> $primitive { self.sum() }
        }
    };
}
impl_simd_wrapper!(packed_simd::f64x8, f64, 8);
impl_simd_wrapper!(packed_simd::f32x8, f32, 8);


struct AlignedFloatVec;
impl AlignedFloatVec {
    pub fn new<T: Primitive>(size: usize) -> Vec<T> {
        use std::{mem, alloc::{alloc_zeroed, Layout}};

        assert_eq!(size % LANES, 0);
        let layout = Layout::from_size_align(size * std::mem::size_of::<T>(), LANES * std::mem::size_of::<T>())
            .expect("Illegal aligned allocation");
        unsafe {
            let aligned_ptr = alloc_zeroed(layout) as *mut T;
            let resvec = Vec::from_raw_parts(aligned_ptr, size, size);
            debug_assert_eq!((resvec.get_unchecked(0) as *const T).align_offset(LANES * std::mem::size_of::<T>()), 0);
            resvec
        }
    }
}



#[derive(Clone, Debug)]
pub struct KMeanState<T: Primitive> {
    pub k: usize,
    pub distsum: T,
    pub centroids: Vec<T>,
    pub centroid_frequency: Vec<usize>,
    pub assignments: Vec<usize>,
    pub centroid_distances: Vec<T>,

    sample_dims: usize,
}
impl<T: Primitive> KMeanState<T> {
    pub fn new(sample_cnt: usize, sample_dims: usize, k: usize) -> Self {
        Self {
            k,
            distsum: T::zero(),
            sample_dims,
            centroids: AlignedFloatVec::new(sample_dims * k),
            centroid_frequency: vec![0usize;k],
            assignments: vec![0usize;sample_cnt],
            centroid_distances: vec![T::infinity();sample_cnt]
        }
    }
    pub fn set_centroid_from_iter(&mut self, idx: usize, src: impl Iterator<Item = T>) {
        self.centroids.iter_mut().skip(self.sample_dims * idx).take(self.sample_dims)
                .zip(src)
                .for_each(|(c,s)| *c = s);
    }

    pub fn remove_padding(mut self, sample_dims: usize) -> Self {
        if self.sample_dims != sample_dims { // Datastructure was padded -> undo
            self.centroids = self.centroids.chunks_exact(self.sample_dims)
                .map(|chunk| chunk.iter().cloned().take(sample_dims)).flatten().collect();
        }
        self
    }
}

fn multiple_roundup(val: usize, multiple_of: usize) -> usize {
    if val % multiple_of != 0 {
        val + multiple_of - (val % multiple_of)
    } else {
        val
    }
}

pub struct KMeans<T> where T: Primitive, [T;LANES]: SimdArray, Simd<[T;LANES]>: SimdWrapper<T> {
    sample_cnt: usize,
    sample_dims: usize,
    p_sample_dims: usize,
    p_samples: Vec<T>
}
impl<T> KMeans<T> where T: Primitive, [T;LANES]: SimdArray, Simd<[T;LANES]>: SimdWrapper<T> {
    pub fn new(samples: Vec<T>, sample_cnt: usize, sample_dims: usize) -> Self {
        assert!(samples.len() == sample_cnt * sample_dims);
        let p_sample_dims = multiple_roundup(sample_dims, LANES);
       
        // Recopy into new, properly aligned + padded buffer
        let mut aligned_samples = AlignedFloatVec::new(sample_cnt * p_sample_dims);
        if p_sample_dims == sample_dims {
            aligned_samples.copy_from_slice(&samples);
        } else {
            for s in 0..sample_cnt {
                for d in 0..sample_dims {
                    aligned_samples[s * p_sample_dims + d] = samples[s * sample_dims + d];
                }
            }
        };

        Self {
            sample_cnt: sample_cnt,
            sample_dims: sample_dims,
            p_sample_dims,
            p_samples: aligned_samples
        }
    }

    fn update_cluster_assignments(&self, state: &mut KMeanState<T>, limit_k: Option<usize>) {
        let centroid_distances = &mut state.centroid_distances;
        let assignments = &mut state.assignments;
        let centroids = &mut state.centroids;

        let k = limit_k.unwrap_or(state.k);

        self.p_samples.par_chunks_exact(self.p_sample_dims)
            .zip(assignments.par_iter_mut())
            .zip(centroid_distances.par_iter_mut())
            .for_each(|((s, assignment), centroid_dist)| {
                let (best_idx, best_dist) = centroids.chunks_exact(self.p_sample_dims).take(k)
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

    fn update_centroids(&self, state: &mut KMeanState<T>) -> T {
        let chunks_per_sample = self.p_sample_dims / LANES;
        // Sum all samples in a cluster together into new_centroids
        // Count non-empty clusters
        let mut used_centroids_cnt = 0;
        state.centroid_frequency.iter_mut().for_each(|v| *v = 0);
        let mut new_centroids = AlignedFloatVec::new(state.centroids.len());
        let mut new_distsum = T::zero();

        let (centroid_frequency, assignments, centroid_distances) = (&mut state.centroid_frequency, &state.assignments, &state.centroid_distances);
        rayon::scope(|s| {
            s.spawn(|_| {
                assignments.iter().cloned()
                    .for_each(|centroid_id| {
                        if centroid_frequency[centroid_id] == 0 {
                            used_centroids_cnt += 1; // Count the amount of centroids with more than 0 samples
                        }
                        centroid_frequency[centroid_id] += 1;
                    });
            });
            s.spawn(|_| {
                self.p_samples.chunks_exact(self.p_sample_dims)
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
            let mut distance_sorted_samples: Vec<usize> = (0..self.sample_cnt).collect();
            distance_sorted_samples.sort_unstable_by(
                |&i1, &i2| state.centroid_distances[i1].partial_cmp(&state.centroid_distances[i2]).unwrap());

            // Assign empty clusters
            for i in 0..state.k {
                if state.centroid_frequency[i] == 0 {
                    // Find the sample with the highest distance to its centroid, that is not alone in its cluster
                    let mut sample_id = std::usize::MAX;
                    let mut centroid_id = std::usize::MAX;
                    for j in self.sample_cnt-1..=0 {
                        sample_id = distance_sorted_samples[j];
                        centroid_id = state.assignments[sample_id];
                        if state.centroid_frequency[centroid_id] > 1 {
                            break;
                        }
                    }
                    // Re-Assign found sample to centroid without any samples
                    let prev_centroid_id = state.assignments[sample_id];
                    state.centroid_frequency[centroid_id] -= 1;
                    state.centroid_frequency[i] += 1;
                    new_distsum -= state.centroid_distances[sample_id];
                    state.centroid_distances[sample_id] = self.p_samples.iter().skip(sample_id * self.p_sample_dims).cloned()
                        .zip(state.centroids.iter().skip(centroid_id * self.p_sample_dims).cloned())
                        .take(self.p_sample_dims)
                        .map(|(s,c)| (s-c) * (s-c))
                        .sum();
                    new_distsum += state.centroid_distances[sample_id];
                    new_centroids.iter_mut().skip(prev_centroid_id * self.p_sample_dims).take(self.p_sample_dims)
                        .zip(self.p_samples.iter().skip(sample_id * self.p_sample_dims).cloned())
                        .for_each(|(cv,sv)| {
                            *cv -= sv;
                        });
                    new_centroids.iter_mut().skip(centroid_id * self.p_sample_dims).take(self.p_sample_dims)
                        .zip(self.p_samples.iter().skip(sample_id * self.p_sample_dims).cloned())
                        .for_each(|(cv,sv)| {
                            *cv += sv;
                        });
                    state.assignments[sample_id] = i;
                }
            }
        }
        // Calculate new centroids from updated cluster_assignments
        state.centroids.chunks_exact_mut(self.p_sample_dims)
            .zip(new_centroids.chunks_exact(self.p_sample_dims))
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

    pub fn kmeans<'a, F>(&self, k: usize, max_iter: usize, init: F, rnd: &'a mut dyn RngCore) -> KMeanState<T>
                where F: FnOnce(&KMeans<T>, &mut KMeanState<T>, &'a mut dyn RngCore) {
        assert!(k <= self.sample_cnt);

        let mut state = KMeanState::new(self.sample_cnt, self.p_sample_dims, k);
        state.distsum = T::infinity();

        // Initialize clusters
        init(&self, &mut state, rnd);

        for _ in 1..=max_iter {
            self.update_cluster_assignments(&mut state, None);

            let new_distsum = self.update_centroids(&mut state);

            if (state.distsum - new_distsum) < T::from(0.0005).unwrap() {
                break;
            }
            state.distsum = new_distsum;
        }

        state.remove_padding(self.sample_dims)
    }

    pub fn init_kmeanplusplus<'a>(kmean: &KMeans<T>, state: &mut KMeanState<T>, rnd: &'a mut dyn RngCore) {
        { // Randomly select first centroid
            let first_idx = rnd.gen_range(0,kmean.sample_cnt);
            state.set_centroid_from_iter(0, kmean.p_samples.iter().skip(first_idx * kmean.p_sample_dims).cloned())
        }
        for k in 1..state.k { // For each following centroid...
            // Calculate distances & update cluster-assignments
            kmean.update_cluster_assignments(state, Some(k));
            
            //NOTE: following two calculations are not what Matlab lists on documentation, but what Matlab actually implemented...
            // Calculate sum of distances per centroid
            let distsum = state.centroid_distances.iter().cloned().sum();

            // Calculate probabilities for each of the samples, to be the new centroid
            let centroid_probabilities: Vec<T> = state.centroid_distances.iter()
                                                    .cloned()
                                                    .map(|d| d / distsum)
                                                    .collect();
            // Use rand's WeightedIndex to randomly draw a centroid, while respecting their probabilities
            let centroid_index = WeightedIndex::new(centroid_probabilities).unwrap();
            let sampled_centroid_id = centroid_index.sample(rnd);
            state.set_centroid_from_iter(k,
                kmean.p_samples.iter().skip(sampled_centroid_id * kmean.p_sample_dims).cloned());
        }
    }

}




#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    macro_rules! assert_approx_eq {
        ($left: expr, $right: expr, $tol: expr) => ({
            match ($left, $right, $tol) {
                (left_val , right_val, tol_val) => {
                    let delta = (left_val - right_val).abs();
                    if !(delta < tol_val) {
                        panic!(
                            "assertion failed: `(left ≈ right)` \
                            (left: `{:?}`, right: `{:?}`) \
                            with ∆={:1.1e} (allowed ∆={:e})",
                            left_val , right_val, delta, tol_val
                        )
                    }
                }
            }
        });
        ($left: expr, $right: expr) => (assert_approx_eq!(($left), ($right), 1e-15))
    }

    #[test]
    fn multiple_roundup() {
        assert_eq!(super::multiple_roundup(0, LANES), 0);
		for i in 1..=LANES {
        	assert_eq!(super::multiple_roundup(i, LANES), LANES);
		}
		for i in LANES+1..=2*LANES {
        	assert_eq!(super::multiple_roundup(i, LANES), 2 * LANES);
		}
    }

    #[test]
    fn padding_and_cluster_assignments() {
        calculate_cluster_assignments_multiplex(1);
        calculate_cluster_assignments_multiplex(2);
        calculate_cluster_assignments_multiplex(3);
        calculate_cluster_assignments_multiplex(97);
        calculate_cluster_assignments_multiplex(98);
        calculate_cluster_assignments_multiplex(99);
        calculate_cluster_assignments_multiplex(100);
    }

    fn calculate_cluster_assignments_multiplex(sample_dims: usize) {
        calculate_cluster_assignments::<f64>(sample_dims, 1e-10f64);
        calculate_cluster_assignments::<f32>(sample_dims, 1e-5f32);
    }

    fn calculate_cluster_assignments<T: Primitive>(sample_dims: usize, max_diff: T) where [T;LANES] : SimdArray, Simd<[T;LANES]>: SimdWrapper<T> {
        let sample_cnt = 1000;
        let k = 5;

        let mut samples = vec![T::zero();sample_cnt * sample_dims];
        samples.iter_mut().for_each(|i| *i = thread_rng().gen_range(T::zero(), T::from(1.0).unwrap()));

        let kmean = KMeans::new(samples, sample_cnt, sample_dims);
        
        let mut state = KMeanState::new(kmean.sample_cnt, kmean.p_sample_dims, k);
        state.centroids.iter_mut()
            .zip(kmean.p_samples.iter())
            .for_each(|(c,s)| *c = *s);

        // calculate distances using method that (hopefully) works.
        let mut should_assignments = state.assignments.clone();
        let mut should_centroid_distances = state.centroid_distances.clone();
        kmean.p_samples.chunks_exact(kmean.p_sample_dims)
            .zip(should_assignments.iter_mut())
            .zip(should_centroid_distances.iter_mut())
            .for_each(|((s, assignment), centroid_dist)| {
                let (best_idx, best_dist) = state.centroids
                    .chunks_exact(kmean.p_sample_dims)
                    .map(|c| {
                        s.iter().cloned().zip(c.iter().cloned())
                            .map(|(sv,cv)| sv - cv)
                            .map(|v| v * v)
                            .sum::<T>()
                    })
                    .enumerate()
                    .min_by(|(_,d0), (_,d1)| d0.partial_cmp(d1).unwrap())
                    .unwrap();
                *assignment = best_idx;
                *centroid_dist = best_dist;
            });

        
        // calculate distances using optimized code
        kmean.update_cluster_assignments(&mut state, None);

        for i in 0..should_assignments.len() {
            assert_approx_eq!(state.centroid_distances[i], should_centroid_distances[i], max_diff);
        }
        assert_eq!(state.assignments, should_assignments);
    }

    #[test]
    fn iris_dataset_f64() where {
        let samples = vec![1.4f64, 0.2, 1.4, 0.2, 1.3, 0.2, 1.5, 0.2, 1.4, 0.2, 1.7, 0.4, 1.4, 0.3, 1.5, 0.2, 1.4, 0.2, 1.5, 0.1, 1.5, 0.2, 1.6, 0.2, 1.4, 0.1, 1.1, 0.1, 1.2, 0.2, 1.5, 0.4, 1.3, 0.4, 1.4, 0.3, 1.7, 0.3, 1.5, 0.3, 1.7, 0.2, 1.5, 0.4, 1.0, 0.2, 1.7, 0.5, 1.9, 0.2, 1.6, 0.2, 1.6, 0.4, 1.5, 0.2, 1.4, 0.2, 1.6, 0.2, 1.6, 0.2, 1.5, 0.4, 1.5, 0.1, 1.4, 0.2, 1.5, 0.2, 1.2, 0.2, 1.3, 0.2, 1.4, 0.1, 1.3, 0.2, 1.5, 0.2, 1.3, 0.3, 1.3, 0.3, 1.3, 0.2, 1.6, 0.6, 1.9, 0.4, 1.4, 0.3, 1.6, 0.2, 1.4, 0.2, 1.5, 0.2, 1.4, 0.2, 4.7, 1.4, 4.5, 1.5, 4.9, 1.5, 4.0, 1.3, 4.6, 1.5, 4.5, 1.3, 4.7, 1.6, 3.3, 1.0, 4.6, 1.3, 3.9, 1.4, 3.5, 1.0, 4.2, 1.5, 4.0, 1.0, 4.7, 1.4, 3.6, 1.3, 4.4, 1.4, 4.5, 1.5, 4.1, 1.0, 4.5, 1.5, 3.9, 1.1, 4.8, 1.8, 4.0, 1.3, 4.9, 1.5, 4.7, 1.2, 4.3, 1.3, 4.4, 1.4, 4.8, 1.4, 5.0, 1.7, 4.5, 1.5, 3.5, 1.0, 3.8, 1.1, 3.7, 1.0, 3.9, 1.2, 5.1, 1.6, 4.5, 1.5, 4.5, 1.6, 4.7, 1.5, 4.4, 1.3, 4.1, 1.3, 4.0, 1.3, 4.4, 1.2, 4.6, 1.4, 4.0, 1.2, 3.3, 1.0, 4.2, 1.3, 4.2, 1.2, 4.2, 1.3, 4.3, 1.3, 3.0, 1.1, 4.1, 1.3, 6.0, 2.5, 5.1, 1.9, 5.9, 2.1, 5.6, 1.8, 5.8, 2.2, 6.6, 2.1, 4.5, 1.7, 6.3, 1.8, 5.8, 1.8, 6.1, 2.5, 5.1, 2.0, 5.3, 1.9, 5.5, 2.1, 5.0, 2.0, 5.1, 2.4, 5.3, 2.3, 5.5, 1.8, 6.7, 2.2, 6.9, 2.3, 5.0, 1.5, 5.7, 2.3, 4.9, 2.0, 6.7, 2.0, 4.9, 1.8, 5.7, 2.1, 6.0, 1.8, 4.8, 1.8, 4.9, 1.8, 5.6, 2.1, 5.8, 1.6, 6.1, 1.9, 6.4, 2.0, 5.6, 2.2, 5.1, 1.5, 5.6, 1.4, 6.1, 2.3, 5.6, 2.4, 5.5, 1.8, 4.8, 1.8, 5.4, 2.1, 5.6, 2.4, 5.1, 2.3, 5.1, 1.9, 5.9, 2.3, 5.7, 2.5, 5.2, 2.3, 5.0, 1.9, 5.2, 2.0, 5.4, 2.3, 5.1, 1.8];

        let kmean = KMeans::new(samples, 150, 2);
        let mut rnd = rand::rngs::StdRng::seed_from_u64(1);
        let res = kmean.kmeans(3, 100, KMeans::init_kmeanplusplus, &mut rnd);

        // SHOULD solution
        let should_assignments = vec![1usize, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
        let should_centroid_distances = vec![0.005960000000000026, 0.005960000000000026, 0.028360000000000038, 0.003559999999999977, 0.005960000000000026, 0.08035999999999992, 0.006760000000000042, 0.003559999999999977, 0.005960000000000026, 0.02275999999999996, 0.003559999999999977, 0.021159999999999964, 0.02516000000000001, 0.15236000000000005, 0.07076000000000011, 0.02516000000000002, 0.049960000000000074, 0.006760000000000042, 0.0595599999999999, 0.004359999999999994, 0.05875999999999988, 0.02516000000000002, 0.21556000000000014, 0.12115999999999995, 0.19395999999999974, 0.021159999999999964, 0.042760000000000006, 0.003559999999999977, 0.005960000000000026, 0.021159999999999964, 0.021159999999999964, 0.02516000000000002, 0.02275999999999996, 0.005960000000000026, 0.003559999999999977, 0.07076000000000011, 0.028360000000000038, 0.02516000000000001, 0.028360000000000038, 0.003559999999999977, 0.029160000000000054, 0.029160000000000054, 0.028360000000000038, 0.14436000000000004, 0.2155599999999998, 0.006760000000000042, 0.021159999999999964, 0.005960000000000026, 0.003559999999999977, 0.005960000000000026, 0.18889053254437893, 0.07812130177514806, 0.4227366863905332, 0.07427514792899402, 0.134275147928994, 0.05504437869822485, 0.2519674556213022, 1.0565828402366864, 0.11119822485207079, 0.13965976331360952, 0.7088905325443784, 0.029659763313609536, 0.18965976331360923, 0.18889053254437893, 0.44965976331360924, 0.020428994082840376, 0.07812130177514806, 0.1458136094674555, 0.07812130177514806, 0.19504437869822466, 0.4911982248520712, 0.07427514792899402, 0.4227366863905332, 0.20581360946745575, 0.002736686390532506, 0.020428994082840376, 0.28504437869822474, 0.4689236111111097, 0.07812130177514806, 0.7088905325443784, 0.2788905325443786, 0.4411982248520705, 0.15658284023668634, 0.4372569444444434, 0.07812130177514806, 0.11965976331360972, 0.21042899408284055, 0.01889053254437878, 0.030428994082840305, 0.07427514792899402, 0.03735207100591719, 0.11273668639053239, 0.09273668639053242, 1.0565828402366864, 0.006582840236686325, 0.025044378698224738, 0.006582840236686325, 0.002736686390532506, 1.6696597633136092, 0.030428994082840305, 0.3772569444444456, 0.2647569444444437, 0.096423611111112, 0.056423611111110925, 0.06809027777777829, 1.0122569444444458, 0.18119822485207124, 0.5522569444444454, 0.09809027777777793, 0.4680902777777788, 0.24725694444444377, 0.10642361111111055, 0.01309027777777764, 0.35642361111110993, 0.377256944444444, 0.15642361111111072, 0.0655902777777774, 1.2455902777777805, 1.769756944444448, 0.5588905325443789, 0.07975694444444478, 0.48559027777777586, 1.2205902777777804, 0.5405902777777757, 0.014756944444444746, 0.21975694444444505, 0.4911982248520712, 0.5405902777777757, 0.003923611111111172, 0.23309027777777774, 0.27309027777777833, 0.6480902777777799, 0.02642361111111129, 0.5347569444444434, 0.40642361111111075, 0.32309027777777855, 0.1314236111111113, 0.0655902777777774, 0.4911982248520712, 0.042256944444443965, 0.1314236111111113, 0.3147569444444439, 0.2647569444444437, 0.16142361111111203, 0.22475694444444502, 0.22559027777777696, 0.37392361111110983, 0.15809027777777682, 0.107256944444444, 0.3022569444444436];
        let should_centroids = vec![4.269230769230769, 1.342307692307692, 1.4620000000000002, 0.2459999999999999, 5.595833333333332, 2.0374999999999996];

        assert_eq!(res.distsum, 31.371358974358966);
        assert_eq!(res.sample_dims, LANES);
        assert_eq!(res.assignments, should_assignments);
        assert_eq!(res.centroid_distances, should_centroid_distances);
        assert_eq!(res.centroids, should_centroids);
    }

    #[test]
    fn iris_dataset_f32() where {
        let samples = vec![1.4f32, 0.2, 1.4, 0.2, 1.3, 0.2, 1.5, 0.2, 1.4, 0.2, 1.7, 0.4, 1.4, 0.3, 1.5, 0.2, 1.4, 0.2, 1.5, 0.1, 1.5, 0.2, 1.6, 0.2, 1.4, 0.1, 1.1, 0.1, 1.2, 0.2, 1.5, 0.4, 1.3, 0.4, 1.4, 0.3, 1.7, 0.3, 1.5, 0.3, 1.7, 0.2, 1.5, 0.4, 1.0, 0.2, 1.7, 0.5, 1.9, 0.2, 1.6, 0.2, 1.6, 0.4, 1.5, 0.2, 1.4, 0.2, 1.6, 0.2, 1.6, 0.2, 1.5, 0.4, 1.5, 0.1, 1.4, 0.2, 1.5, 0.2, 1.2, 0.2, 1.3, 0.2, 1.4, 0.1, 1.3, 0.2, 1.5, 0.2, 1.3, 0.3, 1.3, 0.3, 1.3, 0.2, 1.6, 0.6, 1.9, 0.4, 1.4, 0.3, 1.6, 0.2, 1.4, 0.2, 1.5, 0.2, 1.4, 0.2, 4.7, 1.4, 4.5, 1.5, 4.9, 1.5, 4.0, 1.3, 4.6, 1.5, 4.5, 1.3, 4.7, 1.6, 3.3, 1.0, 4.6, 1.3, 3.9, 1.4, 3.5, 1.0, 4.2, 1.5, 4.0, 1.0, 4.7, 1.4, 3.6, 1.3, 4.4, 1.4, 4.5, 1.5, 4.1, 1.0, 4.5, 1.5, 3.9, 1.1, 4.8, 1.8, 4.0, 1.3, 4.9, 1.5, 4.7, 1.2, 4.3, 1.3, 4.4, 1.4, 4.8, 1.4, 5.0, 1.7, 4.5, 1.5, 3.5, 1.0, 3.8, 1.1, 3.7, 1.0, 3.9, 1.2, 5.1, 1.6, 4.5, 1.5, 4.5, 1.6, 4.7, 1.5, 4.4, 1.3, 4.1, 1.3, 4.0, 1.3, 4.4, 1.2, 4.6, 1.4, 4.0, 1.2, 3.3, 1.0, 4.2, 1.3, 4.2, 1.2, 4.2, 1.3, 4.3, 1.3, 3.0, 1.1, 4.1, 1.3, 6.0, 2.5, 5.1, 1.9, 5.9, 2.1, 5.6, 1.8, 5.8, 2.2, 6.6, 2.1, 4.5, 1.7, 6.3, 1.8, 5.8, 1.8, 6.1, 2.5, 5.1, 2.0, 5.3, 1.9, 5.5, 2.1, 5.0, 2.0, 5.1, 2.4, 5.3, 2.3, 5.5, 1.8, 6.7, 2.2, 6.9, 2.3, 5.0, 1.5, 5.7, 2.3, 4.9, 2.0, 6.7, 2.0, 4.9, 1.8, 5.7, 2.1, 6.0, 1.8, 4.8, 1.8, 4.9, 1.8, 5.6, 2.1, 5.8, 1.6, 6.1, 1.9, 6.4, 2.0, 5.6, 2.2, 5.1, 1.5, 5.6, 1.4, 6.1, 2.3, 5.6, 2.4, 5.5, 1.8, 4.8, 1.8, 5.4, 2.1, 5.6, 2.4, 5.1, 2.3, 5.1, 1.9, 5.9, 2.3, 5.7, 2.5, 5.2, 2.3, 5.0, 1.9, 5.2, 2.0, 5.4, 2.3, 5.1, 1.8];

        let kmean = KMeans::new(samples, 150, 2);
        let mut rnd = rand::rngs::StdRng::seed_from_u64(1);
        let res = kmean.kmeans(3, 100, KMeans::init_kmeanplusplus, &mut rnd);

        // SHOULD solution
        let should_assignments = vec![1usize, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 0, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
        let should_centroid_distances = vec![0.0059600007, 0.0059600007, 0.028360017, 0.0035599954, 0.0059600007, 0.080360025, 0.00676001, 0.0035599954, 0.0059600007, 0.022759989, 0.0035599954, 0.02116, 0.025159994, 0.15235998, 0.070759974, 0.025160013, 0.049960032, 0.00676001, 0.05956002, 0.0043600043, 0.05876001, 0.025160013, 0.21556, 0.12116004, 0.19395997, 0.02116, 0.042760015, 0.0035599954, 0.0059600007, 0.02116, 0.02116, 0.025160013, 0.022759989, 0.0059600007, 0.0035599954, 0.070759974, 0.028360017, 0.025159994, 0.028360017, 0.0035599954, 0.029160026, 0.029160026, 0.028360017, 0.14436004, 0.21555999, 0.00676001, 0.02116, 0.0059600007, 0.0035599954, 0.0059600007, 0.1676405, 0.06282582, 0.38875192, 0.08912205, 0.114307255, 0.046529524, 0.22393683, 1.1143073, 0.098010965, 0.15578863, 0.7572701, 0.02838137, 0.2146776, 0.1676405, 0.4831962, 0.013196193, 0.06282582, 0.16615912, 0.06282582, 0.22134417, 0.4517149, 0.08912205, 0.38875192, 0.19134419, 0.0035665375, 0.013196193, 0.25912234, 0.5129684, 0.06282582, 0.7572701, 0.3098628, 0.48023304, 0.17949231, 0.47731638, 0.06282582, 0.10097398, 0.18578866, 0.015048049, 0.04060358, 0.08912205, 0.03689988, 0.09615911, 0.11097388, 1.1143073, 0.012085075, 0.033936903, 0.012085075, 0.0035665375, 1.7380108, 0.04060358, 0.34427163, 0.29862052, 0.07775034, 0.06209856, 0.053402513, 0.9512281, 0.15912215, 0.5155767, 0.09166375, 0.4290541, 0.2790552, 0.12818542, 0.01862004, 0.39427257, 0.40079403, 0.16992417, 0.07731599, 1.1764451, 1.6864456, 0.5202333, 0.069054514, 0.5294899, 1.1555758, 0.5631963, 0.008185137, 0.2012288, 0.4517149, 0.5631963, 0.0034026075, 0.23079431, 0.24644595, 0.6012286, 0.023837328, 0.57688177, 0.4203598, 0.2881847, 0.12470677, 0.07731599, 0.4517149, 0.053837437, 0.12470677, 0.34035927, 0.29862052, 0.13861972, 0.20992392, 0.2451419, 0.41383788, 0.18383783, 0.114706814, 0.33818585];
        let should_centroids = vec![4.2925925, 1.3592592, 1.462, 0.24599996, 5.626087, 2.0478265];

        assert_eq!(res.distsum, 31.412888);
        assert_eq!(res.sample_dims, LANES);
        assert_eq!(res.assignments, should_assignments);
        assert_eq!(res.centroid_distances, should_centroid_distances);
        assert_eq!(res.centroids, should_centroids);
    }

    #[bench]
    fn distance_matrix_calculation_benchmark_f64(b: &mut Bencher) { distance_matrix_calculation_benchmark::<f64>(b); }
    #[bench]
    fn distance_matrix_calculation_benchmark_f32(b: &mut Bencher) { distance_matrix_calculation_benchmark::<f32>(b); }

    fn distance_matrix_calculation_benchmark<T: Primitive>(b: &mut Bencher) where [T;LANES] : SimdArray, Simd<[T;LANES]>: SimdWrapper<T> {
        let sample_cnt = 20000;
        let sample_dims = 2000;
        let k = 5;

        let mut samples = vec![T::zero();sample_cnt * sample_dims];
        samples.iter_mut().for_each(|v| *v = thread_rng().gen_range(T::zero(), T::from(1.0).unwrap()));
        let kmean = KMeans::new(samples, sample_cnt, sample_dims);

        let mut state = KMeanState::new(kmean.sample_cnt, kmean.p_sample_dims, k);
        state.centroids.iter_mut()
            .zip(kmean.p_samples.iter())
            .for_each(|(c,s)| *c = *s);

        b.iter(|| {
            KMeans::update_cluster_assignments(&kmean, &mut state, None);
            state.clone()
        });
    }

    #[bench] fn complete_benchmark_small_f64(b: &mut Bencher) { complete_benchmark::<f64>(b, 200, 2000, 10, 32); }
    #[bench] fn complete_benchmark_mid_f64(b: &mut Bencher) { complete_benchmark::<f64>(b, 2000, 200, 10, 32); }
    #[bench] fn complete_benchmark_big_f64(b: &mut Bencher) { complete_benchmark::<f64>(b, 10000, 8, 10, 32); }
    #[bench] fn complete_benchmark_huge_f64(b: &mut Bencher) { complete_benchmark::<f64>(b, 20000, 256, 1, 32); }

    #[bench] fn complete_benchmark_small_f32(b: &mut Bencher) { complete_benchmark::<f32>(b, 200, 2000, 10, 32); }
    #[bench] fn complete_benchmark_mid_f32(b: &mut Bencher) { complete_benchmark::<f32>(b, 2000, 200, 10, 32); }
    #[bench] fn complete_benchmark_big_f32(b: &mut Bencher) { complete_benchmark::<f32>(b, 10000, 8, 10, 32); }
    #[bench] fn complete_benchmark_huge_f32(b: &mut Bencher) { complete_benchmark::<f32>(b, 20000, 256, 1, 32); }

    fn complete_benchmark<T: Primitive>(b: &mut Bencher, sample_cnt: usize, sample_dims: usize, max_iter: usize, k: usize)
                    where [T;LANES] : SimdArray, Simd<[T;LANES]>: SimdWrapper<T> {
        let mut rnd = rand::rngs::StdRng::seed_from_u64(1337);
        let mut samples = vec![T::zero();sample_cnt * sample_dims];
        samples.iter_mut().for_each(|v| *v = rnd.gen_range(T::zero(), T::from(1.0).unwrap()));
        let kmean = KMeans::new(samples, sample_cnt, sample_dims);
        b.iter(|| {
            kmean.kmeans(k, max_iter, KMeans::init_kmeanplusplus, &mut rnd)
        });
    }
}
