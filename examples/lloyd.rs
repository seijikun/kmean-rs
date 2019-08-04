use kmeans::*;

fn main() {
    let (sample_cnt, sample_dims, k, max_iter) = (20000, 200, 4, 100);

    // Generate some random data
    let mut samples = vec![0.0f64;sample_cnt * sample_dims];
    samples.iter_mut().for_each(|v| *v = rand::random());

    // Calculate kmeans, using kmean++ as initialization-method
    let kmean = KMeans::new(samples, sample_cnt, sample_dims);
    let result = kmean.kmeans_lloyd(k, max_iter, KMeans::init_kmeanplusplus, &KMeansConfig::default());

    println!("Centroids: {:?}", result.centroids);
    println!("Cluster-Assignments: {:?}", result.assignments);
    println!("Error: {}", result.distsum);
}
