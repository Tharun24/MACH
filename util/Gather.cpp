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

/*
// raw - R, B, M
// indices - R, N
// result = N, M
void cgather(float* raw, long* indices, float* result, int R, int B, int N, int M)
{
    for(int idx = 0; idx < R; ++idx)
    {
        const int raw_offset = idx * B * M;
        const int idx_offset = idx * N;
        for(int jdx = 0; jdx < N; ++jdx)
        {
                        const int offset = jdx * M;
            for(int kdx = 0; kdx < M; ++kdx)
            {
                result[offset + kdx] += raw[raw_offset + indices[idx_offset + jdx] * M + kdx];
            }
        }
    }
}

#include <vector>
#include <thread>

void cgather(float* raw, long* indices, float* result, int R, int B, int N, int M)
{
    const int THREADS = 2;
    std::vector<std::thread> thread_list;
    for(int tdx = 0; tdx < THREADS; ++tdx)
    {
        std::thread t([=] { retrieve(raw, indices, result, R, B, N, M, tdx, THREADS); });
        thread_list.emplace_back(std::move(t));
    }

    for(auto& t : thread_list)
    {
        t.join();
    }
}

void retrieve(float* raw, long* indices, float* result, int R, int B, int N, int M, int tdx, int THREADS)
{
    for(int idx = tdx; idx < R; idx+=THREADS)
    {
        const int raw_offset = idx * B * M;
        const int idx_offset = idx * N;
        for(int jdx = 0; jdx < N; ++jdx)
        {
            const int offset = jdx * M;
            for(int kdx = 0; kdx < M; ++kdx)
            {
                //result[offset + kdx] += raw[raw_offset + indices[idx_offset + jdx] * M + kdx];
                float update = raw[raw_offset + indices[idx_offset + jdx] * M + kdx];
                                std::atomic::fetch_add<float>(result[offset + kdx], update);
            }
        }
    }
}
*/

/*
// Batch Size = 1
void cgather(float* raw, long* indices, float* result, int R, int B, int N, int M)
{
    for(int idx = 0; idx < R; ++idx)
    {
        const int raw_offset = idx * B;
        const int idx_offset = idx * N;
        for(int jdx = 0; jdx < N; ++jdx)
        {
            for(int kdx = 0; kdx < M; ++kdx)
            {
                result[jdx] += raw[raw_offset + indices[idx_offset + jdx]];
            }
        }
    }
}
*/
