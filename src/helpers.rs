pub(crate) fn multiple_roundup(val: usize, multiple_of: usize) -> usize {
    if val % multiple_of != 0 {
        val + multiple_of - (val % multiple_of)
    } else {
        val
    }
}

#[allow(unused_macros)]
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



#[cfg(test)]
mod tests {
	#[test]
    fn multiple_roundup() {
		for o in 1..20 {
			assert_eq!(super::multiple_roundup(0, o), 0);
			for i in 1..=o {
				assert_eq!(super::multiple_roundup(i, o), o);
			}
			for i in o+1..=2*o {
				assert_eq!(super::multiple_roundup(i, o), 2 * o);
			}
		}
    }
}