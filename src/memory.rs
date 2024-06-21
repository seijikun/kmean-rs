use std::{fmt::{Display, LowerExp}, iter, ops, simd::{num::SimdFloat, LaneCount, Simd, SimdElement, SupportedLaneCount}};
use rand::distributions::uniform::SampleUniform;

pub trait Primitive: 'static + SimdElement + SampleUniform
    + Default + Display + LowerExp + Send + Sync + iter::Sum
    + num::Float
    + ops::Add<Output = Self> + ops::Sub<Output = Self> + ops::Mul<Output = Self> + ops::Div<Output = Self>
    + ops::AddAssign + ops::SubAssign
    + for<'a> ops::AddAssign<&'a Self> + for<'a> ops::Sub<&'a Self> {}

impl<T> Primitive for T
where T: 'static + SimdElement + SampleUniform
    + Default + Display + LowerExp + Send + Sync + iter::Sum
    + num::Float
    + ops::Add<Output = Self> + ops::Sub<Output = Self> + ops::Mul<Output = Self> + ops::Div<Output = Self>
    + ops::AddAssign + ops::SubAssign
    + for<'a> ops::AddAssign<&'a Self> + for<'a> ops::Sub<&'a Self> {}

// ##################################################################

pub trait SupportedSimdArray<T: Primitive, const LANES: usize> :
    ops::Sub<Output = Self> + ops::Add<Output = Self> + ops::Mul<Output = Self> + ops::Div<Output = Self>
    + iter::Sum + SimdFloat<Scalar = T> {}

impl<T: Primitive, const LANES: usize> SupportedSimdArray<T, LANES> for Simd<T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: ops::Sub<Output = Simd<T, LANES>>
        + ops::Add<Output = Simd<T, LANES>>
        + ops::Mul<Output = Simd<T, LANES>>
        + ops::Div<Output = Simd<T, LANES>>
        + iter::Sum
        + SimdFloat<Scalar = T> {}

// ##################################################################

pub(crate) struct AlignedFloatVec<const LANES: usize>;
impl <const LANES: usize> AlignedFloatVec<LANES> {
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