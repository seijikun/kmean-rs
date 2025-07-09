use crate::memory::*;
use crate::AbortStrategy;
use core::simd::{LaneCount, Simd, SupportedLaneCount};
use rand::prelude::*;
use rayon::prelude::*;
use std::cell::RefCell;

pub type InitDoneCallbackFn<'a, T> = &'a dyn Fn(&KMeansState<T>);
pub type IterationDoneCallbackFn<'a, T> = &'a dyn Fn(&KMeansState<T>, usize, T);

/// This is a structure holding various configuration options for the a k-means calculations, such as
/// the random number generator to use, or a couple of callbacks, that can be set to get status information from
/// a running k-means calculation.
///
/// For a more detailed information about all possible options, have a look at [`KMeansConfigBuilder`].
pub struct KMeansConfig<'a, T: Primitive> {
    /// Callback that is called, when the initialization phase finished
    /// ## Arguments
    /// - **state**: Current [`KMeansState`] after the initialization
    pub(crate) init_done: InitDoneCallbackFn<'a, T>,
    /// Callback that is called after each iteration
    /// ## Arguments
    /// - **state**: Current[`KMeansState`] after the iteration
    /// - **iteration_id**: Number of the current iteration
    /// - **distsum**: New distance sum (**state** contains the distsum from the previous iteration)
    pub(crate) iteration_done: IterationDoneCallbackFn<'a, T>,
    /// Random number generator to use
    pub(crate) rnd: Box<RefCell<dyn RngCore>>,
    /// The abort-strategy to use for the running calculation
    pub(crate) abort_strategy: AbortStrategy<T>,
}
impl<T: Primitive> Default for KMeansConfig<'_, T> {
    fn default() -> Self {
        Self {
            init_done: &|_| {},
            iteration_done: &|_, _, _| {},
            rnd: Box::new(RefCell::new(rand::thread_rng())),
            abort_strategy: AbortStrategy::<T>::NoImprovement {
                threshold: T::from(0.0005).unwrap(),
            },
        }
    }
}
impl<'a, T: Primitive> KMeansConfig<'a, T> {
    /// Use the [`KMeansConfigBuilder`] to build a [`KMeansConfig`] instance.
    pub fn build() -> KMeansConfigBuilder<'a, T> {
        KMeansConfigBuilder {
            config: KMeansConfig::default(),
        }
    }
}
impl<T: Primitive> std::fmt::Debug for KMeansConfig<'_, T> {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { Ok(()) }
}

pub struct KMeansConfigBuilder<'a, T: Primitive> {
    config: KMeansConfig<'a, T>,
}
impl<'a, T: Primitive> KMeansConfigBuilder<'a, T> {
    /// Set the callback that should be called after the centroid initialization, before the iteration starts.
    pub fn init_done(mut self, init_done: InitDoneCallbackFn<'a, T>) -> Self {
        self.config.init_done = init_done;
        self
    }
    /// Set the callback that should be called after each iteration during a running k-means calculation.
    pub fn iteration_done(mut self, iteration_done: IterationDoneCallbackFn<'a, T>) -> Self {
        self.config.iteration_done = iteration_done;
        self
    }
    /// Set the random number generator that should be used in the k-means calculation.
    /// Use a seeded generator for deterministically repeatable results.
    pub fn random_generator<R: RngCore + 'static>(mut self, rnd: R) -> Self {
        self.config.rnd = Box::new(RefCell::new(rnd));
        self
    }
    /// Set the abort-strategy to use during a running k-means calculation. For more information,
    /// see documentation of [`AbortStrategy`].
    /// ## Default
    /// [`AbortStrategy::NoImprovement`] `{ threshold: 0.0005 }`
    pub fn abort_strategy(mut self, abort_strategy: AbortStrategy<T>) -> Self {
        self.config.abort_strategy = abort_strategy;
        self
    }
    /// Return the internally built configuration structure.
    pub fn build(self) -> KMeansConfig<'a, T> { self.config }
}

/// This is the internally used data-structure, storing the current state during calculation, as
/// well as the final result, as returned by the API.
/// All mutations are done in this structure, making [`KMeans`] immutable, and therefore allowing
/// it to be used in parallel, without having to duplicate the input-data.
///
/// ## Generics
/// - **T**: Underlying primitive type that was used for the calculation
///
/// ## Fields
/// - **k**: The amount of clusters that were requested when calculating this k-means result
/// - **distsum**: The total sum of (squared) distances from all samples to their respective centroids
/// - **centroids**: Calculated cluster centers [row-major] = [<centroid0>,<centroid1>,<centroid2>,...]
/// - **centroid_frequency**: Amount of samples in each centroid
/// - **assignments**: Vector mapping each sample to its respective nearest cluster
/// - **centroid_distances**: Vector containing each sample's (squared) distance to its centroid
#[derive(Clone, Debug)]
pub struct KMeansState<T: Primitive> {
    pub k: usize,
    pub distsum: T,
    pub centroids: StrideBuffer<T>,
    pub centroid_frequency: Vec<usize>,
    pub assignments: Vec<usize>,
    pub centroid_distances: Vec<T>,
}
impl<T: Primitive> KMeansState<T> {
    pub(crate) fn new<const LANES: usize>(sample_cnt: usize, sample_dims: usize, k: usize) -> Self {
        Self {
            k,
            distsum: T::zero(),
            centroids: StrideBuffer::new::<LANES>(k, sample_dims),
            centroid_frequency: vec![0usize; k],
            assignments: vec![0usize; sample_cnt],
            centroid_distances: vec![T::infinity(); sample_cnt],
        }
    }
}

/// A trait representing a customizable distance function for k-means clustering.
///
/// This trait allows you to define your own distance metric to be used in
/// the k-means clustering algorithm.
///
/// # Generics
/// - `T`: The data type of the elements in the slices. This type must support
///        arithmetic operations
/// - `LANES`: SIMD lane size as requested by the API user
pub trait DistanceFunction<T, const LANES: usize>: Send + Sync {
    fn distance(&self, a: &[T], b: &[T]) -> T;
}

/// Entrypoint of this crate's API-Surface.
///
/// Create an instance of this struct, giving the samples you want to operate on. The primitive type
/// of the passed samples array will be the type used internaly for all calculations, as well as the result
/// as stored in the returned [`KMeansState`] structure.
///
/// ## Supported variants
/// - k-Means clustering (Lloyd) [`KMeans::kmeans_lloyd`]
/// - Mini-Batch k-Means clustering [`KMeans::kmeans_minibatch`]
///
/// ## Supported initialization methods
/// - K-Mean++ [`KMeans::init_kmeanplusplus`]
/// - Random-Sample [`KMeans::init_random_sample`]
/// - Random-Partition [`KMeans::init_random_partition`]
///
/// # Generics
/// - `T`: The type of primitive to work with (e.g. f32 of f64)
/// - `LANES`: The amount of SIMD lanes (values in one SIMD vector) to limit the generated code to
///   Note that the generated code selects the appropriate instructions for every platform
/// - `D`: The distance function to use. Default is Euclidean distance.
pub struct KMeans<T, const LANES: usize, D: DistanceFunction<T, LANES>>
where
    T: Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: SupportedSimdArray<T, LANES>,
{
    pub(crate) sample_cnt: usize,
    pub(crate) sample_dims: usize,
    pub(crate) p_samples: StrideBuffer<T>,
    pub(crate) distance_fn: D,
}
impl<T, const LANES: usize, D: DistanceFunction<T, LANES>> KMeans<T, LANES, D>
where
    T: Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: SupportedSimdArray<T, LANES>,
{
    /// Create a new instance of the [`KMeans`] structure.
    ///
    /// ## Arguments
    /// - **samples**: Slice of samples [row-major] = [<sample0>,<sample1>,<sample2>,...]
    /// - **sample_cnt**: Amount of samples, contained in the passed **samples** vector
    /// - **sample_dims**: Amount of dimensions each sample from the **sample** vector has
    /// - **distance_fn**: Distance function to use for the calculation
    pub fn new(samples: &[T], sample_cnt: usize, sample_dims: usize, distance_fn: D) -> Self {
        assert!(samples.len() == sample_cnt * sample_dims);

        Self {
            sample_cnt,
            sample_dims,
            p_samples: StrideBuffer::from_slice::<LANES>(sample_dims, samples),
            distance_fn,
        }
    }

    pub(crate) fn update_centroid_distances(&self, state: &mut KMeansState<T>) {
        let centroids = &state.centroids;

        // manually calculate work-packet size, because rayon does not do static scheduling (which is more apropriate here)
        let work_packet_size = self.p_samples.bfr.len() / self.p_samples.stride / rayon::current_num_threads();
        self.p_samples
            .bfr
            .par_chunks_exact(self.p_samples.stride)
            .with_min_len(work_packet_size)
            .zip(state.assignments.par_iter().cloned())
            .zip(state.centroid_distances.par_iter_mut())
            .for_each(|((s, assignment), centroid_dist)| {
                *centroid_dist = self.distance_fn.distance(s, centroids.nth_stride(assignment));
            });
    }

    pub(crate) fn update_cluster_assignments(&self, state: &mut KMeansState<T>, limit_k: Option<usize>) {
        let centroids = &state.centroids;
        let k = limit_k.unwrap_or(state.k);

        // manually calculate work-packet size, because rayon does not do static scheduling (which is more apropriate here)
        let work_packet_size = self.p_samples.bfr.len() / self.p_samples.stride / rayon::current_num_threads();
        self.p_samples
            .bfr
            .par_chunks_exact(self.p_samples.stride)
            .with_min_len(work_packet_size)
            .zip(state.assignments.par_iter_mut())
            .zip(state.centroid_distances.par_iter_mut())
            .for_each(|((s, assignment), centroid_dist)| {
                let (best_idx, best_dist) = centroids
                    .chunks_exact_stride()
                    .take(k)
                    .map(|c| self.distance_fn.distance(s, c))
                    .enumerate()
                    .min_by(|(_, d0), (_, d1)| d0.partial_cmp(d1).unwrap())
                    .unwrap();
                *assignment = best_idx;
                *centroid_dist = best_dist;
            });
    }

    pub(crate) fn update_cluster_frequencies(&self, assignments: &[usize], centroid_frequency: &mut [usize]) -> usize {
        centroid_frequency.iter_mut().for_each(|v| *v = 0);
        let mut used_centroids_cnt = 0;
        assignments.iter().cloned().for_each(|centroid_id| {
            if centroid_frequency[centroid_id] == 0 {
                used_centroids_cnt += 1; // Count the amount of centroids with more than 0 samples
            }
            centroid_frequency[centroid_id] += 1;
        });
        used_centroids_cnt
    }

    /// Normal K-Means algorithm implementation. This is the same algorithm as implemented in Matlab (one-phase).
    /// (see: https://uk.mathworks.com/help/stats/kmeans.html#bueq7aj-5    Section: More About)
    ///
    /// ## Arguments
    /// - **k**: Amount of clusters to search for
    /// - **max_iter**: Limit the maximum amount of iterations (just pass a high number for infinite)
    /// - **init**: Initialization-Method to use for the initialization of the **k** centroids
    /// - **config**: [`KMeansConfig`] instance, containing several configuration options for the calculation.
    ///
    /// ## Returns
    /// Instance of [`KMeansState`], containing the final state (result).
    ///
    /// ## Example
    /// ```rust
    /// use kmeans::*;
    ///
    /// let (sample_cnt, sample_dims, k, max_iter) = (20000, 200, 4, 100);
    ///
    /// // Generate some random data
    /// let mut samples = vec![0.0f64;sample_cnt * sample_dims];
    /// samples.iter_mut().for_each(|v| *v = rand::random());
    ///
    /// // Calculate kmeans, using kmean++ as initialization-method
    /// // KMeans<_, 8> specifies to use f64 SIMD vectors with 8 lanes (e.g. AVX512)
    /// let kmean: KMeans<_, 8, _> = KMeans::new(&samples, sample_cnt, sample_dims, EuclideanDistance);
    /// let result = kmean.kmeans_lloyd(k, max_iter, KMeans::init_kmeanplusplus, &KMeansConfig::default());
    ///
    /// println!("Centroids: {:?}", result.centroids);
    /// println!("Cluster-Assignments: {:?}", result.assignments);
    /// println!("Error: {}", result.distsum);
    /// ```
    pub fn kmeans_lloyd<F>(&self, k: usize, max_iter: usize, init: F, config: &KMeansConfig<'_, T>) -> KMeansState<T>
    where
        for<'c> F: FnOnce(&KMeans<T, LANES, D>, &mut KMeansState<T>, &KMeansConfig<'c, T>),
    {
        crate::variants::Lloyd::calculate(self, k, max_iter, init, config)
    }

    /// Mini-Batch k-Means implementation.
    /// (see: https://dl.acm.org/citation.cfm?id=1772862)
    ///
    /// ## Arguments
    /// - **batch_size**: Amount of samples to use per iteration (higher -> better approximation but slower)
    /// - **k**: Amount of clusters to search for
    /// - **max_iter**: Limit the maximum amount of iterations (just pass a high number for infinite)
    /// - **init**: Initialization-Method to use for the initialization of the **k** centroids
    /// - **config**: [`KMeansConfig`] instance, containing several configuration options for the calculation.
    ///
    /// ## Returns
    /// Instance of [`KMeansState`], containing the final state (result).
    ///
    /// ## Example
    /// ```rust
    /// use kmeans::*;
    ///
    /// let (sample_cnt, sample_dims, k, max_iter) = (20000, 200, 4, 100);
    ///
    /// // Generate some random data
    /// let mut samples = vec![0.0f64;sample_cnt * sample_dims];
    /// samples.iter_mut().for_each(|v| *v = rand::random());
    ///
    /// // Calculate kmeans, using kmean++ as initialization-method
    /// // KMeans<_, 8> specifies to use f64 SIMD vectors with 8 lanes (e.g. AVX512)
    /// let kmean: KMeans<_, 8, _> = KMeans::new(&samples, sample_cnt, sample_dims, EuclideanDistance);
    /// let result = kmean.kmeans_minibatch(4, k, max_iter, KMeans::init_random_sample, &KMeansConfig::default());
    ///
    /// println!("Centroids: {:?}", result.centroids);
    /// println!("Cluster-Assignments: {:?}", result.assignments);
    /// println!("Error: {}", result.distsum);
    /// ```
    pub fn kmeans_minibatch<F>(&self, batch_size: usize, k: usize, max_iter: usize, init: F, config: &KMeansConfig<'_, T>) -> KMeansState<T>
    where
        for<'c> F: FnOnce(&KMeans<T, LANES, D>, &mut KMeansState<T>, &KMeansConfig<'c, T>),
        T: Primitive,
        LaneCount<LANES>: SupportedLaneCount,
        Simd<T, LANES>: SupportedSimdArray<T, LANES>,
    {
        crate::variants::Minibatch::calculate(self, batch_size, k, max_iter, init, config)
    }

    /// K-Means++ initialization method, as implemented in Matlab
    ///
    /// ## Description
    /// This initialization method starts by selecting one sample as first centroid.
    /// Proceeding from there, the method iteratively selects one new centroid (per iteration) by calculating
    /// each sample's probability of "being a centroid". This probability is bigger, the farther away a sample
    /// is from its centroid. Then, one sample is randomly selected, while taking their probability of being
    /// the next centroid into account. This leads to a tendency of selecting centroids, that are far away from
    /// their currently assigned cluster's centroid.
    /// (see: https://uk.mathworks.com/help/stats/kmeans.html#bueq7aj-5    Section: More About)
    ///
    /// ## Note
    /// This method is not meant for direct invocation. Pass a reference to it, to an instance-method of [`KMeans`].
    pub fn init_kmeanplusplus(kmean: &KMeans<T, LANES, D>, state: &mut KMeansState<T>, config: &KMeansConfig<'_, T>) {
        crate::inits::kmeanplusplus::calculate(kmean, state, config);
    }

    /// Random-Parition initialization method
    ///
    /// ## Description
    /// This initialization method randomly partitions the samples into k partitions, and then calculates these partion's means.
    /// These means are then used as initial clusters.
    ///
    pub fn init_random_partition(kmean: &KMeans<T, LANES, D>, state: &mut KMeansState<T>, config: &KMeansConfig<'_, T>) {
        crate::inits::randompartition::calculate(kmean, state, config);
    }

    /// Random sample initialization method (a.k.a. Forgy)
    ///
    /// ## Description
    /// This initialization method randomly selects k centroids from the samples as initial centroids.
    ///
    /// ## Note
    /// This method is not meant for direct invocation. Pass a reference to it, to an instance-method of [`KMeans`].
    pub fn init_random_sample(kmean: &KMeans<T, LANES, D>, state: &mut KMeansState<T>, config: &KMeansConfig<'_, T>) {
        crate::inits::randomsample::calculate(kmean, state, config);
    }

    /// Precomputed centroids initialization method
    ///
    /// ## Description
    /// This initialization method requires a precomputed list of k centroids to use as initial
    /// centroids.
    ///
    /// ## Note
    /// This method must be invoked with a precomputed list of centroids. It then
    /// returns a closure that can be passed to the [`KMeans`] object.
    pub fn init_precomputed(centroids: Vec<T>) -> impl Fn(&KMeans<T, LANES, D>, &mut KMeansState<T>, &KMeansConfig<'_, T>) {
        move |kmean, state, config| {
            crate::inits::precomputed::calculate(kmean, state, config, &centroids);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EuclideanDistance;
    use test::Bencher;

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
        calculate_cluster_assignments::<f64, 8>(sample_dims, 1e-10f64);
        calculate_cluster_assignments::<f64, 4>(sample_dims, 1e-10f64);
        calculate_cluster_assignments::<f64, 2>(sample_dims, 1e-10f64);

        calculate_cluster_assignments::<f32, 16>(sample_dims, 1.2e-5f32);
        calculate_cluster_assignments::<f32, 8>(sample_dims, 1.2e-5f32);
        calculate_cluster_assignments::<f32, 4>(sample_dims, 1.2e-5f32);
        calculate_cluster_assignments::<f32, 2>(sample_dims, 1.2e-5f32);
    }

    fn calculate_cluster_assignments<T, const LANES: usize>(sample_dims: usize, max_diff: T)
    where
        T: Primitive,
        LaneCount<LANES>: SupportedLaneCount,
        Simd<T, LANES>: SupportedSimdArray<T, LANES>,
    {
        let sample_cnt = 1000;
        let k = 5;

        let mut samples = vec![T::zero(); sample_cnt * sample_dims];
        let mut rng = rand::rngs::StdRng::seed_from_u64(1337);
        samples.iter_mut().for_each(|i| *i = rng.gen_range(T::zero()..T::one()));

        let kmean = KMeans::new(&samples, sample_cnt, sample_dims, EuclideanDistance);

        let mut state = KMeansState::new::<LANES>(kmean.sample_cnt, sample_dims, k);
        state
            .centroids
            .bfr
            .iter_mut()
            .zip(kmean.p_samples.bfr.iter())
            .for_each(|(c, s)| *c = *s);

        // calculate distances using method that (hopefully) works.
        let mut should_assignments = state.assignments.clone();
        let mut should_centroid_distances = state.centroid_distances.clone();
        kmean
            .p_samples
            .chunks_exact_stride()
            .zip(should_assignments.iter_mut())
            .zip(should_centroid_distances.iter_mut())
            .for_each(|((s, assignment), centroid_dist)| {
                let (best_idx, best_dist) = state
                    .centroids
                    .chunks_exact_stride()
                    .map(|c| {
                        s.iter()
                            .cloned()
                            .zip(c.iter().cloned())
                            .map(|(sv, cv)| sv - cv)
                            .map(|v| v * v)
                            .sum::<T>()
                    })
                    .enumerate()
                    .min_by(|(_, d0), (_, d1)| d0.partial_cmp(d1).unwrap())
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

    #[bench]
    fn distance_matrix_calculation_benchmark_f64x8(b: &mut Bencher) { distance_matrix_calculation_benchmark::<f64, 8>(b); }
    #[bench]
    fn distance_matrix_calculation_benchmark_f64x4(b: &mut Bencher) { distance_matrix_calculation_benchmark::<f64, 4>(b); }
    #[bench]
    fn distance_matrix_calculation_benchmark_f64x2(b: &mut Bencher) { distance_matrix_calculation_benchmark::<f64, 2>(b); }

    #[bench]
    fn distance_matrix_calculation_benchmark_f32x16(b: &mut Bencher) { distance_matrix_calculation_benchmark::<f32, 16>(b); }
    #[bench]
    fn distance_matrix_calculation_benchmark_f32x8(b: &mut Bencher) { distance_matrix_calculation_benchmark::<f32, 8>(b); }
    #[bench]
    fn distance_matrix_calculation_benchmark_f32x4(b: &mut Bencher) { distance_matrix_calculation_benchmark::<f32, 4>(b); }
    #[bench]
    fn distance_matrix_calculation_benchmark_f32x2(b: &mut Bencher) { distance_matrix_calculation_benchmark::<f32, 2>(b); }

    fn distance_matrix_calculation_benchmark<T, const LANES: usize>(b: &mut Bencher)
    where
        T: Primitive,
        LaneCount<LANES>: SupportedLaneCount,
        Simd<T, LANES>: SupportedSimdArray<T, LANES>,
    {
        let sample_cnt = 20000;
        let sample_dims = 2000;
        let k = LANES;

        let mut samples = vec![T::zero(); sample_cnt * sample_dims];
        let mut rng = rand::rngs::StdRng::seed_from_u64(1337);
        samples.iter_mut().for_each(|v| *v = rng.gen_range(T::zero()..T::one()));
        let kmean: KMeans<T, LANES, _> = KMeans::new(&samples, sample_cnt, sample_dims, EuclideanDistance);

        let mut state = KMeansState::new::<LANES>(kmean.sample_cnt, sample_dims, k);
        state
            .centroids
            .bfr
            .iter_mut()
            .zip(kmean.p_samples.bfr.iter())
            .for_each(|(c, s)| *c = *s);

        b.iter(|| {
            KMeans::update_cluster_assignments(&kmean, &mut state, None);
            state.clone()
        });
    }
}
