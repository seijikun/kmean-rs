use crate::helpers;
use core::fmt;
use rand::distributions::uniform::SampleUniform;
use std::ops::{Index, IndexMut};
use std::simd::num::SimdFloat;
use std::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};
use std::{iter, ops};

pub trait Primitive:
    'static
    + SimdElement
    + SampleUniform
    + Default
    + fmt::Display
    + fmt::Debug
    + fmt::LowerExp
    + Send
    + Sync
    + iter::Sum
    + num::Float
    + ops::Add<Output = Self>
    + ops::Sub<Output = Self>
    + ops::Mul<Output = Self>
    + ops::Div<Output = Self>
    + ops::AddAssign
    + ops::SubAssign
    + for<'a> ops::AddAssign<&'a Self>
    + for<'a> ops::Sub<&'a Self>
{
}

impl<T> Primitive for T where
    T: 'static
        + SimdElement
        + SampleUniform
        + Default
        + fmt::Display
        + fmt::Debug
        + fmt::LowerExp
        + Send
        + Sync
        + iter::Sum
        + num::Float
        + ops::Add<Output = Self>
        + ops::Sub<Output = Self>
        + ops::Mul<Output = Self>
        + ops::Div<Output = Self>
        + ops::AddAssign
        + ops::SubAssign
        + for<'a> ops::AddAssign<&'a Self>
        + for<'a> ops::Sub<&'a Self>
{
}

// ##################################################################

pub trait SupportedSimdArray<T: Primitive, const LANES: usize>:
    ops::Sub<Output = Self> + ops::Add<Output = Self> + ops::Mul<Output = Self> + ops::Div<Output = Self> + iter::Sum + SimdFloat<Scalar = T>
{
}

impl<T: Primitive, const LANES: usize> SupportedSimdArray<T, LANES> for Simd<T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: ops::Sub<Output = Simd<T, LANES>>
        + ops::Add<Output = Simd<T, LANES>>
        + ops::Mul<Output = Simd<T, LANES>>
        + ops::Div<Output = Simd<T, LANES>>
        + iter::Sum
        + SimdFloat<Scalar = T>,
{
}

// ##################################################################

pub(crate) type SIMDAlignBuffer<T> = aligned_vec::AVec<T, aligned_vec::RuntimeAlign>;
pub(crate) struct SIMDAlignBufferHelper;
impl SIMDAlignBufferHelper {
    pub fn create<T: Primitive, const LANES: usize>(size: usize) -> SIMDAlignBuffer<T> {
        aligned_vec::avec_rt!([ LANES * std::mem::size_of::<T>() ]| Default::default(); size)
    }
}

// ##################################################################

/// Buffer storing strided multi-dimensional elements, like centroids.
///
/// # Details
/// kmeans internally uses SIMD instructions. To do that efficiently, it is easiest
/// to pad all handled data to multiples of the used LANES size. This datastructure
/// internally handles what is padding and what is not, and allows easy access to
/// the contained (padded) elements for users.
#[derive(Clone, Debug)]
pub struct StrideBuffer<T> {
    pub(crate) bfr: SIMDAlignBuffer<T>,
    pub(crate) stride: usize,
    pub(crate) centroid_cnt: usize,
    pub(crate) centroid_dim: usize,
}
impl<T: Primitive> StrideBuffer<T> {
    pub fn new<const LANES: usize>(centroid_cnt: usize, centroid_dim: usize) -> Self {
        let stride = helpers::multiple_roundup(centroid_dim, LANES);
        Self {
            bfr: SIMDAlignBufferHelper::create::<T, LANES>(stride * centroid_cnt),
            stride,
            centroid_cnt,
            centroid_dim,
        }
    }

    pub fn from_slice<const LANES: usize>(centroid_dim: usize, vec: &[T]) -> Self {
        assert_eq!(vec.len() % centroid_dim, 0);
        let stride = helpers::multiple_roundup(centroid_dim, LANES);
        let centroid_cnt = vec.len() / centroid_dim;

        let mut bfr = Self {
            bfr: SIMDAlignBufferHelper::create::<T, LANES>(stride * centroid_cnt),
            stride,
            centroid_cnt,
            centroid_dim,
        };
        for i in 0..centroid_cnt {
            bfr.set_nth_from_iter(i, vec[i * centroid_dim..].iter().cloned());
        }
        bfr
    }

    /// Convert the centroids contained in this CentroidBuffer to a tightly packed
    /// Vec with the memory layout `[c0_0, c0_1, ... c0_n, c1_0, c1_1, ... c1_n, ...]`
    /// where `cx_y` is the `y`'th dimension of the `x`'th centroid.
    pub fn to_vec(&self) -> Vec<T> {
        let mut res = Vec::with_capacity(self.centroid_cnt * self.centroid_dim);
        res.extend(self.iter().flat_map(|c| c.iter()));
        res
    }

    /// Set the nth stride from the given iterator
    pub fn set_nth_from_iter(&mut self, index: usize, iter: impl IntoIterator<Item = T>) {
        self.index_mut(index).iter_mut().zip(iter).for_each(|(dst, src)| {
            *dst = src;
        });
    }

    /// Iterate over all contained centroids. Every item in the iterator
    /// corresponds to one centroid.
    pub fn iter(&self) -> impl Iterator<Item = &[T]> { self.bfr.chunks_exact(self.stride).map(|c| &c[..self.centroid_dim]) }

    /// Iterate over all contained centroids. Every item in the iterator
    /// corresponds to one centroid.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
        self.bfr.chunks_exact_mut(self.stride).map(|c| &mut c[..self.centroid_dim])
    }

    // #############################################
    // INTERNAL

    #[inline(always)]
    pub(crate) fn nth_stride(&self, index: usize) -> &[T] {
        let offset = index * self.stride;
        &self.bfr[offset..offset + self.stride]
    }

    #[inline(always)]
    pub(crate) fn nth_stride_mut(&mut self, index: usize) -> &mut [T] {
        let offset = index * self.stride;
        &mut self.bfr[offset..offset + self.stride]
    }

    #[inline(always)]
    pub(crate) fn chunks_exact_stride(&self) -> impl Iterator<Item = &[T]> { self.bfr.chunks_exact(self.stride) }

    #[inline(always)]
    pub(crate) fn chunks_exact_stride_mut(&mut self) -> impl Iterator<Item = &mut [T]> { self.bfr.chunks_exact_mut(self.stride) }
}
impl<T> Index<usize> for StrideBuffer<T> {
    type Output = [T];

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        let offset = index * self.stride;
        &self.bfr[offset..offset + self.centroid_dim]
    }
}
impl<T> IndexMut<usize> for StrideBuffer<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let offset = index * self.stride;
        &mut self.bfr[offset..offset + self.centroid_dim]
    }
}

#[cfg(test)]
mod tests {
    use super::StrideBuffer;

    #[test]
    fn stride_buffer() {
        for centroid_dim in 1..8 {
            let bfr: StrideBuffer<f32> = StrideBuffer::new::<8>(5, centroid_dim);
            assert_eq!(bfr.stride, 8);
            assert_eq!(bfr.bfr.alignment(), 8 * std::mem::size_of::<f32>());
            assert_eq!(bfr.centroid_dim, centroid_dim);
            assert_eq!(bfr.centroid_cnt, 5);
        }

        {
            let mut bfr: StrideBuffer<f64> = StrideBuffer::new::<8>(3, 9);
            assert_eq!(bfr.stride, 16);
            assert_eq!(bfr.bfr.alignment(), 8 * std::mem::size_of::<f64>());
            assert_eq!(bfr.centroid_dim, 9);
            assert_eq!(bfr.centroid_cnt, 3);
            bfr.set_nth_from_iter(1, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
            assert_eq!(bfr.bfr.to_vec(), vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]);
        }

        {
            let src_bfr = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
            let bfr: StrideBuffer<f32> = StrideBuffer::from_slice::<4>(3, &src_bfr);
            assert_eq!(bfr.centroid_cnt, 4);
            assert_eq!(bfr.centroid_dim, 3);
            assert_eq!(bfr.stride, 4);
            assert_eq!(bfr.bfr.to_vec(), vec![
                1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0, 7.0, 8.0, 9.0, 0.0, 10.0, 11.0, 12.0, 0.0,
            ]);
            assert_eq!(bfr.to_vec(), src_bfr);
        }
    }
}
