#include <queue>
#include "Gather.h"
#include <iostream>
using namespace std;
// raw - M, R, B
// indices - R, N
// result = M, N
void cgather_batch(float* raw, long* lookup, float* result, long* top_preds, int R, int B, int N, int batch_size, int n_threads)
{
    vector<priority_queue<pair<float, long>>> q(batch_size);
    #pragma omp parallel for num_threads(n_threads)   
    for(int idx = 0; idx < batch_size; ++idx)
    {
        const int preds_offset = idx * 5;
        const int scores_offset = idx * N;    
        for(int rdx = 0; rdx < R; ++rdx)
        {
            const int idx_offset = rdx * N;
            const int raw_offset = idx * R * B + rdx * B;

            for(int kdx = 0; kdx < N; ++kdx)
            {
                result[scores_offset + kdx] += raw[raw_offset + lookup[idx_offset + kdx]];
            }
        }
        // filling the queue
        for(long i = scores_offset; i < scores_offset+N; ++i)
        {
            if(q[idx].size()<5)
                q[idx].push(pair<float, long>(-result[i], i));
            else if(q[idx].top().first > -result[i]){
                q[idx].pop();
                q[idx].push(pair<float, long>(-result[i], i));
            }    
        }
        // getting the top 5 classes
        for(long i = preds_offset; i < preds_offset+5; ++i)
        {
            top_preds[i] = q[idx].top().second;
            q[idx].pop();
        }
    }
}

void cgather_K(float* raw, long* lookup, float* result, long* top_preds, int R, int B, int N, int batch_size, int n_threads)
{       
    vector<priority_queue<pair<float, long>>> q(batch_size);  
    for(int idx = 0; idx < batch_size; ++idx)
    {
        const int preds_offset = idx * 5;
        const int scores_offset = idx * N;    
        for(int rdx = 0; rdx < R; ++rdx)
        {
            const int idx_offset = rdx * N;
            const int raw_offset = idx * R * B + rdx * B;

            #pragma omp parallel for num_threads(n_threads) 
            for(int kdx = 0; kdx < N; ++kdx)
            {
                result[scores_offset + kdx] += raw[raw_offset + lookup[idx_offset + kdx]];
            }
        }
        // filling the queue
        for(long i = scores_offset; i < scores_offset+N; ++i)
        {
            if(q[idx].size()<5)
                q[idx].push(pair<float, long>(-result[i], i));
            else if(q[idx].top().first > -result[i]){
                q[idx].pop();
                q[idx].push(pair<float, long>(-result[i], i));
            }    
        }
        // getting the top 5 classes
        for(long i = preds_offset; i < preds_offset+5; ++i)
        {
            top_preds[i] = q[idx].top().second;
            q[idx].pop();
        }
    }
}

