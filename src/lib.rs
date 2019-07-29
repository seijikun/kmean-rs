#![feature(test)]
extern crate test;

mod helpers;
mod memory;
mod kmeans_impl;

pub use kmeans_impl::{KMeanState, KMeans};



#[cfg(test)]
mod tests {
    use test::Bencher;
    use rand::prelude::*;
    use super::*;
    use crate::memory::*;
    use packed_simd::{Simd, SimdArray};

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
