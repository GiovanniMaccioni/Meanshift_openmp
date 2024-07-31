#include "include/functions.cpp"

int main()
{
    dataset2D data;
    dataset2D centroids;
    
    std::string file_csv_in;
    std::string file_csv_out;
    std::string dataset_path;

    dataset_path = "./dataset/blobs_5clusters_100000samples";
    file_csv_out = file_csv_in + "_output.csv";
    file_csv_in = file_csv_in+".csv";

    data.x = (float*)malloc(sizeof(float)*DIM_DATASET);
    data.y = (float*)malloc(sizeof(float)*DIM_DATASET);
    data.labels = (int*)malloc(sizeof(int)*DIM_DATASET);


    upload_dataset2D(data, dataset_path+file_csv_in);

    //Here we initialize the centroid data structure, that will be modified throughout the algorithm
    //For starters the centroids are all the points in the dataset
    centroids.x = copy_array(data.x, DIM_DATASET);
    centroids.y = copy_array(data.y, DIM_DATASET);
    centroids.labels = new int[DIM_DATASET];
    /*Centroid Labels inizialization*/
    for(int i = 0; i < DIM_DATASET; i++)
        centroids.labels[i] = -1;


    auto start_conv = std::chrono::high_resolution_clock::now();

    int nthreads = 0;//TOCHECK

    /*Uncomment for M1; Comment for M2
    
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        #ifdef _OPENMP
            int tid = omp_get_thread_num();
            if (tid == 0)
            {
                nthreads = omp_get_num_threads();
                //printf("Number of threads = %d\n", nthreads);
            }
        #endif

        #pragma omp for
        for(int i = 0; i < DIM_DATASET; ++i)
        {
            for(int j = 0; j < MAX_ITERS; ++j)
                meanshift2D(centroids, data, i, DIM_DATASET, BANDWIDTH);
        }
    }
    
    #ifdef _OPENMP
        printf("Number of threads = %d\n", nthreads);
    #endif*/

    //Uncomment for M2; Comment for M1
    for(int i = 0; i < DIM_DATASET; ++i)
    {
        for(int j = 0; j < MAX_ITERS; ++j)
            meanshift2D(centroids, data, i, DIM_DATASET, BANDWIDTH);
    }
    //Uncomment for M2; Comment for M1
    

    auto stop_conv = std::chrono::high_resolution_clock::now();
    auto duration_conv = std::chrono::duration_cast<std::chrono::milliseconds>(stop_conv - start_conv);

    

    printf("Work took %ld milliseconds\n", duration_conv.count());

    auto start = std::chrono::high_resolution_clock::now();

    merge_cluster2D(centroids, DIM_DATASET, BANDWIDTH);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    printf("Work took %ld milliseconds\n", duration.count());

    for(int i = 0; i < DIM_DATASET; i++)
        data.labels[i] = centroids.labels[i];

    write_csv(data, DIM_DATASET, dataset_path+file_csv_out);
}