use crate::memory::*;

/// Enum with possible abort strategies.
/// These strategies specify when a running iteration (with the k-means calculation) is aborted.
pub enum AbortStrategy<T: Primitive> {
	/// This strategy aborts the calculation directly after an iteration produced no improvement where `improvement > threshold`
	/// for the first time.
	/// ## Fields:
	/// - **threshold**: Threshold, used to detect an improvement (`improvement > threshold`)
    NoImprovement { threshold: T },
    /// This strategy aborts the calculation, when there have not been any improvements after **x** iterations,
    /// where `improvement > threshold`.
	/// ## Fields:
	/// - **x**: The amount of consecutive without improvement, after which the calculation is aborted
	/// - **threshold**: Threshold, used to detect an improvement (`improvement > threshold`)
	/// - **abort_on_negative**: Specifies whether the strategy instantly aborts when a negative improvement occured (**true**), or if
	/// negative improvements are handled as "no improvements" (**false**).
	NoImprovementForXIterations { x: usize, threshold: T, abort_on_negative: bool }
}
impl<T: Primitive> AbortStrategy<T> {
	pub(crate) fn create_logic(&self) -> Box<dyn AbortStrategyLogic<T>> {
		match *self {
			AbortStrategy::NoImprovementForXIterations{x,threshold,abort_on_negative} => Box::new(NoImprovementForXIterationsLogic {
				x, threshold, abort_on_negative,
				prev_error: T::infinity(),
				no_improvement_counter: 0
			}),
			AbortStrategy::NoImprovement{threshold} => Box::new(NoImprovementLogic {
				threshold,
				prev_error: T::infinity()
			})
		}
	}
}

pub(crate) trait AbortStrategyLogic<T: Primitive> {
	/// Function that has to be called once an iteration of the calculation ended, a new error was calculated.
	/// ## Arguments
	/// - **error**: The new **error (distsum), after an iteration
	/// ## Returns
	/// - **true** if the calculation should continue
	/// - **false** if the calculation should abort
	fn next(&mut self, error: T) -> bool;
}


pub(crate) struct NoImprovementLogic<T: Primitive> {
	threshold: T,
	prev_error: T
}
impl<T: Primitive> AbortStrategyLogic<T> for NoImprovementLogic<T> {
	fn next(&mut self, error: T) -> bool {
		let improvement = self.prev_error - error;
		self.prev_error = error;
		improvement > self.threshold
	}
}


pub(crate) struct NoImprovementForXIterationsLogic<T: Primitive> {
	x: usize,
	threshold: T,
	abort_on_negative: bool,
	prev_error: T,
	no_improvement_counter: usize
}
impl<T: Primitive> AbortStrategyLogic<T> for NoImprovementForXIterationsLogic<T> {
	fn next(&mut self, error: T) -> bool {
		let improvement = self.prev_error - error;
		self.prev_error = error;
		if self.abort_on_negative && improvement < T::zero() { // Negative improvement, and instant abort is requested
			return false;
		}
		if improvement > self.threshold { // positive improvement: reset no-improv-counter
			self.no_improvement_counter = 0;
		} else { // Still no improvement, count 1 up
			self.no_improvement_counter += 1;
		}
		self.no_improvement_counter < self.x
	}
}


#[cfg(test)]
mod tests {
	use super::*;

	#[test] fn test_no_improvement_f32() { test_no_improvement::<f32>(); }
	#[test] fn test_no_improvement_f64() { test_no_improvement::<f64>(); }

	fn test_no_improvement<T: Primitive>() {
		{
			let mut abort_strategy = AbortStrategy::NoImprovement { threshold: T::from(0.0005).unwrap() }.create_logic();
			assert_eq!(abort_strategy.next( T::from(3000.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(3000.0).unwrap() ), false);
		}
		{
			let mut abort_strategy = AbortStrategy::NoImprovement { threshold: T::from(0.0005).unwrap() }.create_logic();
			assert_eq!(abort_strategy.next( T::from(3000.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(2999.99959).unwrap() ), false);
		}
		{
			let mut abort_strategy = AbortStrategy::NoImprovement { threshold: T::from(0.0005).unwrap() }.create_logic();
			assert_eq!(abort_strategy.next( T::from(3000.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(2999.99935).unwrap() ), true);
		}
		{
			let mut abort_strategy = AbortStrategy::NoImprovement { threshold: T::from(0.0005).unwrap() }.create_logic();
			assert_eq!(abort_strategy.next( T::from(3000.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(2000.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(1999.99).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(1999.99999999).unwrap() ), false);
		}
	}


	#[test]
	fn test_no_improvement_for_x_iterations_f32() { test_no_improvement_for_x_iterations::<f32>(); }

	#[test]
	fn test_no_improvement_for_x_iterations_f64() { test_no_improvement_for_x_iterations::<f64>(); }

	fn test_no_improvement_for_x_iterations<T: Primitive>() {
		{
			let mut abort_strategy = AbortStrategy::NoImprovementForXIterations {
				x: 1, threshold: T::from(0.0005).unwrap(), abort_on_negative: false}.create_logic();
			assert_eq!(abort_strategy.next( T::from(3000.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(3000.0).unwrap() ), false);
		}
		{
			let mut abort_strategy = AbortStrategy::NoImprovementForXIterations {
				x: 1, threshold: T::from(0.0005).unwrap(), abort_on_negative: false}.create_logic();
			assert_eq!(abort_strategy.next( T::from(3000.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(2999.99959).unwrap() ), false);
		}
		{
			let mut abort_strategy = AbortStrategy::NoImprovementForXIterations {
				x: 1, threshold: T::from(0.0005).unwrap(), abort_on_negative: false}.create_logic();
			assert_eq!(abort_strategy.next( T::from(3000.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(2999.99935).unwrap() ), true);
		}
		{
			let mut abort_strategy = AbortStrategy::NoImprovementForXIterations {
				x: 1, threshold: T::from(0.0005).unwrap(), abort_on_negative: false}.create_logic();
			assert_eq!(abort_strategy.next( T::from(3000.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(2000.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(1999.99).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(1999.99999999).unwrap() ), false);
		}
		// ABORT_ON_NEGATIVE (without negative improvements)
		{
			let mut abort_strategy = AbortStrategy::NoImprovementForXIterations {
				x: 1, threshold: T::from(0.0005).unwrap(), abort_on_negative: true}.create_logic();
			assert_eq!(abort_strategy.next( T::from(3000.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(3000.0).unwrap() ), false);
		}
		{
			let mut abort_strategy = AbortStrategy::NoImprovementForXIterations {
				x: 1, threshold: T::from(0.0005).unwrap(), abort_on_negative: true}.create_logic();
			assert_eq!(abort_strategy.next( T::from(3000.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(2999.99959).unwrap() ), false);
		}
		{
			let mut abort_strategy = AbortStrategy::NoImprovementForXIterations {
				x: 1, threshold: T::from(0.0005).unwrap(), abort_on_negative: true}.create_logic();
			assert_eq!(abort_strategy.next( T::from(3000.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(2999.99935).unwrap() ), true);
		}
		{
			let mut abort_strategy = AbortStrategy::NoImprovementForXIterations {
				x: 1, threshold: T::from(0.0005).unwrap(), abort_on_negative: true}.create_logic();
			assert_eq!(abort_strategy.next( T::from(3000.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(2000.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(1999.99).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(1999.99999999).unwrap() ), false);
		}
		// ABORT_ON_NEGATIVE (with negative improvements)
		{
			let mut abort_strategy = AbortStrategy::NoImprovementForXIterations {
				x: 2, threshold: T::from(0.0005).unwrap(), abort_on_negative: true}.create_logic();
			assert_eq!(abort_strategy.next( T::from(3000.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(3001.0).unwrap() ), false);
		}
		{ // Should abort on negative improvement, even ifs absolute value < threshold
			let mut abort_strategy = AbortStrategy::NoImprovementForXIterations {
				x: 2, threshold: T::from(0.0005).unwrap(), abort_on_negative: true}.create_logic();
			assert_eq!(abort_strategy.next( T::from(3000.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(3000.0004).unwrap() ), false);
		}
		{
			let mut abort_strategy = AbortStrategy::NoImprovementForXIterations {
				x: 2, threshold: T::from(0.0005).unwrap(), abort_on_negative: true}.create_logic();
			assert_eq!(abort_strategy.next( T::from(3000.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(3000.0007).unwrap() ), false);
		}

		// X != 1
		{
			let mut abort_strategy = AbortStrategy::NoImprovementForXIterations {
				x: 2, threshold: T::from(0.0005).unwrap(), abort_on_negative: false}.create_logic();
			assert_eq!(abort_strategy.next( T::from(3000.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(2000.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(2000.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(1999.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(1999.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(1999.0).unwrap() ), false);
		}
		{ // Same as directly above, but with abort_on_negative = true
			let mut abort_strategy = AbortStrategy::NoImprovementForXIterations {
				x: 2, threshold: T::from(0.0005).unwrap(), abort_on_negative: true}.create_logic();
			assert_eq!(abort_strategy.next( T::from(3000.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(2000.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(2000.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(1999.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(1999.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(1999.0).unwrap() ), false);
		}
		{ // Negative improvement before no_improvement_counter == 2
			let mut abort_strategy = AbortStrategy::NoImprovementForXIterations {
				x: 2, threshold: T::from(0.0005).unwrap(), abort_on_negative: true}.create_logic();
			assert_eq!(abort_strategy.next( T::from(3000.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(2000.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(2000.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(1999.0).unwrap() ), true);
			assert_eq!(abort_strategy.next( T::from(2999.0).unwrap() ), false);
		}
	}
}