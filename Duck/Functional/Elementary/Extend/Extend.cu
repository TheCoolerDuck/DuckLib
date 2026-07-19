
#include "Functional\\GPUCode\\GPU_Matrix.h"

extern "C" __global__ void Main(Matrix m, Matrix r)
{
    int ID = blockIdx.x * blockDim.x + threadIdx.x;

    int2 quards = r.Quards(ID);

    int x = quards.x;
    int y = quards.y;

    if (ID < r.width * r.height)
    {
        int mx = m.width == 1 ? 0 : x;
        int my = m.height == 1 ? 0 : y;

        r.Add(x, y, m.Get(mx, my));
    }
}