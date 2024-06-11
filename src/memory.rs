use num::{NumCast, Zero, Float};
use std::{
    fmt::{Debug, Display, LowerExp}, iter::Sum, ops::{Add, AddAssign, Sub, SubAssign}, simd::{num::SimdFloat, LaneCount, SupportedLaneCount}
};
use rand::distributions::uniform::SampleUniform;

pub trait Primitive: Add + AddAssign + Sum + Sub + SubAssign + Zero + Float + NumCast + SampleUniform
                + PartialOrd + Copy + Default + Display + Debug + Sync + Send + LowerExp + 'static
                + for<'a> AddAssign<&'a Self> + for<'a> Sub<&'a Self> {}
impl Primitive for f32 {}
impl Primitive for f64 {}


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