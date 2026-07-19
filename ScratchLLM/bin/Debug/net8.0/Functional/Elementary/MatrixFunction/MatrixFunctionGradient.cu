
#include "Functional\\GPUCode\\GPU_Matrix.h"

//a must be the same size as result
extern "C" __global__ void Main(Matrix a, Matrix b, Matrix ag, Matrix bg, Matrix rg, int funcID)
{
    int ID = blockIdx.x * blockDim.x + threadIdx.x;

    int2 quards = a.Quards(ID);

    int x = quards.x;
    int y = quards.y;

    if (ID < a.width * a.height)
    {
        float2 grads = apply(a.Get(x, y), b.Get(x, y), funcID);

        float grad = rg.Get(x, y);

        ag.Add(x, y, grads.x * grad);
        bg.Add(x, y, grads.y * grad);
    }
}