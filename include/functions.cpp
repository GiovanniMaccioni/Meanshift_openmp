#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include "math.h"
#include <omp.h>
#include <chrono>
#include <string.h>


#define MAX_ITERS 15
#define BANDWIDTH 1.25
#define DIM_DATASET 100000
#define NUM_THREADS 32
/*------------------------------------------*/

/*Data structure to memorize datasets with two features*/
struct dataset2D
{
    float* x;
    float* y;
    int* labels;
};

/*---------------------DATA AND MEMORY MANAGING FUNCTIONS-------------------------------*/
//Function to copy an array.
//TOCHECK stiamo parlando di talvolta dimensione campionarie molto alte. La copia non so quanto possa essere performante e/o efficace
inline float* copy_array(float* target, int dim)
{
    float* copy = (float*)malloc(sizeof(float)*dim);
    memcpy(copy, target, sizeof(float)*dim);
    return copy;  
}


//Function to upload the dataset in memory
void upload_dataset2D(dataset2D &d, std::string fname)
{
	std::string line, word;
 
	std::fstream file (fname, std::ios::in);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << fname << std::endl;
        return;
    }
    int counter = 0;
    while(std::getline(file, line, '\n'))
    {
        std::stringstream str(line);
        int component = 0;

        while(std::getline(str, word, ','))
        {
            if(component == 0)
                d.x[counter] = std::stof(word);
            else if(component == 1)
                d.y[counter] = std::stof(word);
            else
                d.labels[counter] = std::stoi(word);
        
            component++;
            //std::cout << stof(word) << "\t";
        }
        //std::cout << std::endl;
        counter++;
    }
}

//Function to write the output
void write_csv(dataset2D &d, int o_d_dim, std::string fname)
{
    std::fstream file (fname, std::ios::out);
    
    for(int i = 0; i < o_d_dim; i++)
        file << d.x[i] << ',' << d.y[i] << ',' << d.labels[i] << '\n';

    file.close();
}

void print_dataset2D(dataset2D d, int d_dim)
{
    for(int i = 0; i < d_dim; i++)
    {
        std::cout << d.x[i]<< "\t" << d.y[i] << "\t" << d.labels[i] << std::endl;
    }
    std::cout << std::endl;
}

/*---------------------------------------------------------------------------------------------*/

//Funzione per calcolare la distanza in 2 dimensioni

inline float euclidian_norm2D(float x, float y)
{
    return sqrt(x*x + y*y);
}

//Function to verify that, given a bandwidth(circle radius in this case),
//a point is cinsidered a neighbour of another point.

inline bool is_neighbour2D(float c_x, float c_y, float p_x, float p_y, float bandwidth)
{
    return euclidian_norm2D(c_x - p_x, c_y - p_y) < bandwidth? true:false;
}

void meanshift2D(dataset2D &c, dataset2D &o_d, int c_index, int o_d_dim, float bandwidth)
{
    float mean_x = 0.0, mean_y = 0.0;
    int neighbours = 0;
    double distance;

    //Uncomment for M2; Comment for M1 (pragma)
    #pragma omp parallel for num_threads(NUM_THREADS) reduction(+: mean_x, mean_y, neighbours)
    for(int i = 0; i < o_d_dim; i++)
    {
        if(is_neighbour2D(c.x[c_index], c.y[c_index], o_d.x[i], o_d.y[i], bandwidth))
        {
            mean_x += o_d.x[i];
            mean_y += o_d.y[i];
            neighbours++;
        }
    }

    if(neighbours != 0)
    {
        c.x[c_index] = mean_x / neighbours;
        c.y[c_index] = mean_y / neighbours;
    }

    return; 
}

void merge_cluster2D(dataset2D &c, int o_d_dim, float b)
{
    int l_counter = 0;
    for(int c_index = 0; c_index < o_d_dim; c_index++)
    {
        //if the centroid hasn't been altready labeled, assign a label and advance the label counter to start a new cluster
        if(c.labels[c_index] == -1)
        {
            //#pragma omp parallel for num_threads(NUM_THREADS)
            for(int j = 0; j < o_d_dim; j++)
            {
                //if the centroid hasn't been labed already
                if(c.labels[j] == -1)
                {
                    if(is_neighbour2D(c.x[c_index], c.y[c_index], c.x[j], c.y[j], b))
                        c.labels[j] = l_counter;
                }
            }
            l_counter++;
        }
    }
}

//



