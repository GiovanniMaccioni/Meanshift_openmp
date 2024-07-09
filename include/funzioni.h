#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include "math.h"
#include <omp.h>
#include <chrono>


#define THRESHOLD 0.1
#define BANDWIDTH 2
#define DIM_DATASET 10000
#define NUM_THREADS 16
/*------------------------------------------*/

//TODO Ricontrollare se devo mettere i const agli argomenti delle funzioni

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
//TODOFunzione per ridurre la dimensionalità degli array
//Da utilizzare per ridurre il numero di centroidi da controllare al passo successivo dell algoritmo
inline float* reduce_array(float* target, int dim)
{
    float* reduction = (float*)malloc(sizeof(float)*dim);
    return reduction;
}

//TODO aggiungere errore di lettura/scrittura per terminare il programma
//Function to upload the dataset in memory
void upload_dataset2D(dataset2D &d, std::string fname)
{
	std::string line, word;
 
	std::fstream file (fname, std::ios::in);
	if(file.is_open())
	{
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

            //if(file.eof() == 1)
            //    break;
		}
	}
	else
		std::cout<<"Could not open the file\n";
}

//Function to write the output
void write_csv(dataset2D &c, dataset2D &d, int o_d_dim, std::string fname)
{
    std::fstream file (fname, std::ios::out);
    
    for(int i = 0; i < o_d_dim; i++)
        file << d.x[i] << ',' << d.y[i] << ',' << c.labels[i] << '\n';

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
    return sqrt(pow(x,2) + pow(y,2));
}

//Function to verify that, given a bandwidth(circle radius in this case),
//a point is cinsidered a neighbour of another point.
//TOCHECK includere o no l'uguale nell'espressione.

inline bool is_neighbour2D(float c_x, float c_y, float p_x, float p_y, float bandwidth)
{
    //DEBUG
    //std::cout << "distance: " << euclidian_norm2D(c_x - p_x, c_y - p_y)<<std::endl;
    //
    return euclidian_norm2D(c_x - p_x, c_y - p_y) < bandwidth? true:false;
}

//
float meanshift2D(dataset2D &c, dataset2D &o_d, int c_index, int o_d_dim, float bandwidth)
{
    float mean_x = 0.0, mean_y = 0.0; //variables that will contain partial sums
    int neighbours = 0;
    double distance;

    //#pragma omp parallel for reduction(+: mean_x, mean_y, neighbours)
    for(int i = 0; i < o_d_dim; i++)
    {
        bool check = false;
        check = is_neighbour2D(c.x[c_index], c.y[c_index], o_d.x[i], o_d.y[i], bandwidth);
        //DEBUG
        //std::cout << check << std::endl;
        //
        if(check)
        {
            mean_x += o_d.x[i];
            mean_y += o_d.y[i];
            neighbours++;
        }
        check = false;
    }

    //DEBUG
    //std::cout << "mean_x: " << mean_x << std::endl;
    //std::cout << "mean_y: " << mean_y << std::endl;
    //

    //Se neighbours è uguale a 0, probabilmente ciò che abbiamo trovato è un outlier, credo possa capitare
    //soltanto nel passo inziale
    if(neighbours != 0)
    {
        mean_x /= neighbours;
        mean_y /= neighbours;
    }

    //TOCHECKPer adesso se il punto è isolato, viene preso come cluster a se stante. Una soluzione eventuale
    //sarebbe quella di introdurre più livelli di bandwidth, per ridurre il rischio di troppa frammentazione
    //dei cluster
    else 
    {
        mean_x = c.x[c_index];
        mean_y = c.y[c_index];
    }
    //DEBUG
    //std::cout << "mean_x: " << mean_x << std::endl;
    //std::cout << "mean_y: " << mean_y << std::endl;
    //
    distance = euclidian_norm2D(c.x[c_index] - mean_x, c.y[c_index] - mean_y);

    c.x[c_index] = mean_x;
    c.y[c_index] = mean_y;

    return distance; 
}

void merge_cluster2D(dataset2D &c, int o_d_dim, float b)
{
    int l_counter = 0;
    for(int c_index = 0; c_index < o_d_dim; c_index++)
    {
        //DEBUG
        //std::cout << "inside for(int i = 0; i < o_d_dim; i++)" << std::endl;
        //
        //if the centroid hasn't been altready labeled, assign a label and advance the label counter to start a new cluster
        if(c.labels[c_index] == -1)
        {
            //DEBUG
            //std::cout << "inside if(c.labels[i] == -1):" << std::endl;
            //
            c.labels[c_index] = l_counter;
            bool check = false;
            //for every other point in the dataset beyond the centroid
            //  (the points before should have been already labeled)
            for(int j = c_index + 1; j < o_d_dim; j++)
            {
                //if the centroid hasn't been labed already
                if(c.labels[j] == -1)
                {
                    //DEBUG
                    //std::cout << "inside if(c.labels[j] == -1):" << std::endl;
                    //
                    check = is_neighbour2D(c.x[c_index], c.y[c_index], c.x[j], c.y[j], b);
                    if(check == true)
                        c.labels[j] = l_counter;
                }
                check = false;
            }
            l_counter++;
        }
    }
}

void confront_labels(dataset2D &c, dataset2D &o_d, int o_d_dim)
{
    for(int i = 0; i < o_d_dim; i++)
    {
        if(c.labels[i] == o_d.labels[i])
            std::cout << "TRUE" << std::endl;
        else
            std::cout << "FALSE" << std::endl;
    }
    std::cout << std::endl;
}


//



