use kmeans::*;

fn main() {
    let (sample_cnt, sample_dims, k, max_iter) = (20000, 200, 4, 100);

    // Generate some random data
    let mut samples = vec![0.0f64;sample_cnt * sample_dims];
    samples.iter_mut().for_each(|v| *v = rand::random());

	let conf = KMeansConfig::build()
        .abort_strategy(AbortStrategy::NoImprovementForXIterations {
            // Abort after there has not been an improvement for 5 iterations
            x: 5,
            // Only count as improvement if > 0.0005 difference
            threshold: 0.0005f64,
            // Do not directly abort after a negative improvement
            abort_on_negative: false
        })
		.build();

    // Calculate kmeans, using kmean++ as initialization-method
    let kmean = KMeans::new(samples, sample_cnt, sample_dims);
    let result = kmean.kmeans_minibatch(4, k, max_iter, KMeans::init_random_sample, &conf);

    println!("Centroids: {:?}", result.centroids);
    println!("Cluster-Assignments: {:?}", result.assignments);
    println!("Error: {}", result.distsum);
}
