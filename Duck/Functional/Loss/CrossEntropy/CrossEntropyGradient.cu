
#include "Functional\\GPUCode\\GPU_Matrix.h"

extern "C" __global__ void Main(Matrix m, Matrix p, int* i)
{
    int ID = blockIdx.x * blockDim.x + threadIdx.x;

    int2 quards = m.Quards(ID);

    int x = quards.x;
    int y = quards.y;

    if (ID < m.width * m.height)
    {
        m.Add(x, y, y == i[x] ? p.Get(x, y) - 1 : p.Get(x, y));
    }
}