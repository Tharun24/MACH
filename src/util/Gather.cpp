#include "Gather.h"
#include <iostream>
// raw - M, R, B
// indices - R, N
// result = M, N
void cgather_batch(float* raw, long* indices, float* result, int R, int B, int N, int batch_size, int n_threads)
{   
    #pragma omp parallel for num_threads(n_threads)
    for(int idx = 0; idx < batch_size; ++idx)
    {
        const int offset = idx * N;    
        
        for(int jdx = 0; jdx < R; ++jdx)
        {
            const int idx_offset = jdx * N;
            const int raw_offset = idx * R * B + jdx * B;
            
            for(int kdx = 0; kdx < N; ++kdx)
            {
                result[offset + kdx] += raw[raw_offset + indices[idx_offset + kdx]];
            }
        }
    }
}

void cgather_K(float* raw, long* indices, float* result, int R, int B, int N, int batch_size, int n_threads)
{       
    for(int idx = 0; idx < batch_size; ++idx)
    {
        const int offset = idx * N;    
        
        for(int jdx = 0; jdx < R; ++jdx)
        {
            const int idx_offset = jdx * N;
            const int raw_offset = idx * R * B + jdx * B;

            #pragma omp parallel for num_threads(n_threads)
            for(int kdx = 0; kdx < N; ++kdx)
            {
                result[offset + kdx] += raw[raw_offset + indices[idx_offset + kdx]];
            }
        }
    }
}

