#include "include/funzioni.h"

int main()
{
    dataset2D data;
    dataset2D centroids;
    
    std::string file_csv_in;
    std::string file_csv_out;
    std::string dataset_path;

    dataset_path = ".\\dataset\\data10000";
    file_csv_out = file_csv_in + "_output.csv";
    file_csv_in = file_csv_in+".csv";

    data.x = (float*)malloc(sizeof(float)*DIM_DATASET);
    data.y = (float*)malloc(sizeof(float)*DIM_DATASET);
    data.labels = (int*)malloc(sizeof(int)*DIM_DATASET);


    upload_dataset2D(data, dataset_path+file_csv_in);
    //print_dataset2D(data, DIM_DATASET);

    //Here we initialize the centroid data structure, that will be modified throughout the algorithm
    //For starters the centroids are all the points in the dataset
    centroids.x = copy_array(data.x, DIM_DATASET);
    centroids.y = copy_array(data.y, DIM_DATASET);
    centroids.labels = new int[DIM_DATASET];
    /*Centroid Labels inizialization*/
    for(int i = 0; i < DIM_DATASET; i++)
        centroids.labels[i] = -1;

    //std::cout << "------CENTROIDS--------" << std::endl;
    //print_dataset2D(centroids, DIM_DATASET);

    auto start = std::chrono::high_resolution_clock::now();

    int nthreads = 0;//TOCHECK

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
        for(int i = 0; i < DIM_DATASET; i++)
        {
            float epsilon = THRESHOLD + 1;
            while(epsilon > THRESHOLD)//TOCHECK aggiungere un contatore di iterazioni?
            {
                epsilon = meanshift2D(centroids, data, i, DIM_DATASET, BANDWIDTH);
            }
            epsilon = THRESHOLD + 1;
        }
    }
    
    #ifdef _OPENMP
        printf("Number of threads = %d\n", nthreads);
    #endif
    
    //DEBUG
        printf("Updated centroids\n");
    //

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    

    printf("Work took %d milliseconds\n", duration.count());

    //std::cout << "------UPDATED CENTROIDS--------" << std::endl;
    //print_dataset2D(centroids, DIM_DATASET);

    /*
    auto start = std::chrono::high_resolution_clock::now();
    */

    //We supposed that, given epsilon small enough, if c1 is close to c2 and c2 to c3, then c1 is close to c3
    merge_cluster2D(centroids, DIM_DATASET, 2*BANDWIDTH);

    /*auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);*/

    printf("Cluster Merged\n");
    //print_dataset2D(centroids, DIM_DATASET);
    //confront_labels(centroids, data, DIM_DATASET);

    write_csv(centroids, data, DIM_DATASET, dataset_path+file_csv_out);
}