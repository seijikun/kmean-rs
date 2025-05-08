use kmeans::*;

fn main() {
    let (sample_cnt, sample_dims, k, max_iter) = (20000, 200, 4, 2500);

    // Generate some random data
    let mut samples = vec![0.0f64; sample_cnt * sample_dims];
    samples.iter_mut().for_each(|v| *v = rand::random());

    let conf = KMeansConfig::build()
        .init_done(&|_| println!("Initialization completed."))
        .iteration_done(&|s: &KMeansState<f64>, nr: usize, new_distsum: f64| {
            println!(
                "Iteration {} - Error: {:.2} -> {:.2} | Improvement: {:.2}",
                nr,
                s.distsum,
                new_distsum,
                s.distsum - new_distsum
            );
            s.centroids
                .iter()
                .enumerate()
                .for_each(|(centroid_idx, centroid)| println!("Centroid[{}]: {:?}", centroid_idx, centroid));
        })
        .build();

    // Calculate kmeans, using kmean++ as initialization-method
    // KMeans<_, 8> specifies to use f64 SIMD vectors with 8 lanes (e.g. AVX512)
    let kmean: KMeans<f64, 8, _> = KMeans::new(&samples, sample_cnt, sample_dims, EuclideanDistance);
    let result = kmean.kmeans_minibatch(4, k, max_iter, KMeans::init_random_sample, &conf);

    println!("Centroids: {:?}", result.centroids);
    println!("Cluster-Assignments: {:?}", result.assignments);
    println!("Error: {}", result.distsum);
}
