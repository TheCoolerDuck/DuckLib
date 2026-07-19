#include "Functional\\GPUCode\\GPU_Matrix.h"

extern "C" __global__ void Main(Matrix m, int* i, Matrix r)
{
    int ID = blockIdx.x * blockDim.x + threadIdx.x;

    if (ID < r.width)
    {
        r.Set(ID, 0, -logf(m.Get(ID, i[ID]) + 1e-8f));
    }
}