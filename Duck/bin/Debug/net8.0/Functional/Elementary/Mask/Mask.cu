#include "Functional\\GPUCode\\GPU_Matrix.h"

extern "C" __global__ void Main(Matrix m, int* mask, double value, Matrix r)
{
    int ID = blockIdx.x * blockDim.x + threadIdx.x;

    int2 quards = r.Quards(ID);

    int x = quards.x;
    int y = quards.y;

    if (ID < r.width * r.height)
    {
        r.Add(x, y, mask[ID] == 1 ? value : m.Get(x, y));
    }
}