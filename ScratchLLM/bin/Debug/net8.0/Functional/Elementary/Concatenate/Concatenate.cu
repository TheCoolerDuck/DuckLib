
#include "Functional\\GPUCode\\GPU_Matrix.h"

extern "C" __global__ void Main(Matrix* a, Matrix r)
{
    int ID = blockIdx.x * blockDim.x + threadIdx.x;

    int2 quards = r.Quards(ID);

    int x = quards.x;
    int y = quards.y;

    if (ID < r.width * r.height)
    {
        int my = y;

        int i = 0;

        while (my >= a[i].height)
            my -= a[i++].height;
        
        r.Add(x, y, a[i].Get(x, my));
    }
}