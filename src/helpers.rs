pub(crate) fn multiple_roundup(val: usize, multiple_of: usize) -> usize {
    if val % multiple_of != 0 {
        val + multiple_of - (val % multiple_of)
    } else {
        val
    }
}

#[cfg(test)]
macro_rules! assert_approx_eq {
	($left: expr, $right: expr, $tol: expr) => ({
		match ($left, $right, $tol) {
			(left_val , right_val, tol_val) => {
				let delta = (left_val - right_val).abs();
				if !(delta < tol_val) {
					panic!(
						"assertion failed: `(left ≈ right)` \
						(left: `{}`, right: `{}`) \
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
pub(crate) mod testing {
	use std::{collections::HashMap, hash::Hash};

use crate::{KMeansState, Primitive};

	pub struct KMeansShouldResult<T: Primitive> {
		pub distsum: T,
		pub sample_dims: usize,
		pub assignments: Vec<usize>,
		pub centroid_distances: Vec<T>,
		pub centroids: Vec<T>
	}

	pub fn assert_kmeans_result_eq<T: Primitive>(should: KMeansShouldResult<T>, actual: KMeansState<T>) {
		let cmp_epsilon = T::from(0.01).unwrap();
		assert_approx_eq!(should.distsum, actual.distsum, cmp_epsilon);

		// compare cluster assignments - and while doing so, generate sorting indices for the centroids
		let mut should_freq: HashMap<usize, usize> = HashMap::new();
		let mut idmap = HashMap::new();
		let mut idrevmap = HashMap::new();
		for idx in 0..should.assignments.len() {
			let (should_id, actual_id) = (should.assignments[idx], actual.assignments[idx]);
			if !idmap.contains_key(&should_id) {
				assert_eq!(idrevmap.contains_key(&actual_id), false);
				idmap.insert(should_id, actual_id);
				idrevmap.insert(actual_id, should_id);
			}
			if idmap[&should_id] != actual_id {
				panic!(
					"Cluster assignments different at idx {}.\nMapping(should -> actual): {:?}\nActual: {:?}\nShould: {:?}",
					idx, idmap, actual.assignments, should.assignments
				);
			}
			should_freq.insert(actual_id, should_freq.get(&actual_id).cloned().unwrap_or_default() + 1);
		}
		// use idmap to compare should & actual in correct order
		for (should_idx, actual_idx) in idmap {
			//assert_approx_eq!(should.centroid_distances[should_idx], actual.centroid_distances[actual_idx], CMP_EPSILON);
			assert_eq!(should_freq[&actual_idx], actual.centroid_frequency[actual_idx]);
			let should_spl_offset = should_idx * should.sample_dims;
			let actual_spl_offset = actual_idx * should.sample_dims;
			for d in 0..should.sample_dims {
				assert_approx_eq!(should.centroids[should_spl_offset + d], actual.centroids[actual_spl_offset + d], cmp_epsilon);
			}
		}
		for idx in 0..should.centroid_distances.len() {
			let (should_dist, actual_dist) = (should.centroid_distances[idx], actual.centroid_distances[idx]);
			if should_dist.abs_sub(actual_dist) > cmp_epsilon {
				panic!("Centroid distances mismatch at idx {}. Actual: {} but should have been: {}", idx, actual_dist, should_dist);
			}
		}
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