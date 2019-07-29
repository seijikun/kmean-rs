pub(crate) fn multiple_roundup(val: usize, multiple_of: usize) -> usize {
    if val % multiple_of != 0 {
        val + multiple_of - (val % multiple_of)
    } else {
        val
    }
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