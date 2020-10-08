use num::{NumCast, Zero, Float};
use std::{
    ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign},
    fmt::{Display, Debug, LowerExp},
    iter::Sum
};
use rand::distributions::uniform::SampleUniform;
use packed_simd::SimdArray;

// TODO: Remove this and use const_generics, as soon as they are stable and the compiler stops crashing :)
pub(crate) const LANES: usize = 8;

pub trait Primitive: Add + AddAssign + Sum + Sub + SubAssign + Zero + Float + NumCast + SampleUniform
                + PartialOrd + Copy + Default + Display + Debug + Sync + Send + LowerExp + 'static
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
            #[inline(always)] unsafe fn from_slice_aligned_unchecked(src: &[$primitive]) -> Self { <$simd>::from_slice_aligned_unchecked(src) }
            #[inline(always)] unsafe fn write_to_slice_aligned_unchecked(self, slice: &mut [$primitive]) { self.write_to_slice_aligned_unchecked(slice); }
            #[inline(always)] fn splat(single: $primitive) -> Self { <$simd>::splat(single) }
            #[inline(always)] fn sum(self) -> $primitive { self.sum() }
        }
    };
}
impl_simd_wrapper!(packed_simd::f64x8, f64, 8);
impl_simd_wrapper!(packed_simd::f32x8, f32, 8);


pub(crate) struct AlignedFloatVec;
impl AlignedFloatVec {
    pub fn new<T: Primitive>(size: usize) -> Vec<T> {
        use std::alloc::{alloc_zeroed, Layout};

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
    pub fn new_uninitialized<T: Primitive>(size: usize) -> Vec<T> {
        use std::alloc::{alloc, Layout};

        assert_eq!(size % LANES, 0);
        let layout = Layout::from_size_align(size * std::mem::size_of::<T>(), LANES * std::mem::size_of::<T>())
            .expect("Illegal aligned allocation");
        unsafe {
            let aligned_ptr = alloc(layout) as *mut T;
            let resvec = Vec::from_raw_parts(aligned_ptr, size, size);
            debug_assert_eq!((resvec.get_unchecked(0) as *const T).align_offset(LANES * std::mem::size_of::<T>()), 0);
            resvec
        }
    }
}