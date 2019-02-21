void cgather_batch(float*, long*, float*, int, int, int, int, int);
void cgather_K(float*, long*, float*, int, int, int, int, int);
void retrieve(float* raw, long* indices, float* result, int R, int B, int N, int M, int tdx, int THREADS);
