use kmeans::*;

fn main() {
    let (sample_cnt, sample_dims, k, max_iter) = (20000, 200, 4, 2500);

    // Generate some random data
    let mut samples = vec![0.0f64;sample_cnt * sample_dims];
    samples.iter_mut().for_each(|v| *v = rand::random());

	let conf = KMeansConfig::build()
		.init_done(&|_| println!("Initialization completed."))
		.iteration_done(&|s, nr, new_distsum|
			println!("Iteration {} - Error: {:.2} -> {:.2} | Improvement: {:.2}",
				nr, s.distsum, new_distsum, s.distsum - new_distsum))
		.build();

    // Calculate kmeans, using kmean++ as initialization-method
    let kmean = KMeans::new(samples, sample_cnt, sample_dims);
    let result = kmean.kmeans_minibatch(4, k, max_iter, KMeans::init_random_sample, &conf);

    println!("Centroids: {:?}", result.centroids);
    println!("Cluster-Assignments: {:?}", result.assignments);
    println!("Error: {}", result.distsum);
}
