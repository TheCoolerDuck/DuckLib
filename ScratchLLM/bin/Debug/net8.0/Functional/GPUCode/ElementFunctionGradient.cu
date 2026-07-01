
#include "Functional\\GPUCode\\GPU_Matrix.h"
#include "Functional\\GPUCode\\ElementFuntionsApply.h"


extern "C" __global__ void Main(Matrix a, Matrix ag, Matrix bg, int funcID)
{
    int ID = blockIdx.x * blockDim.x + threadIdx.x;

    int2 quards = r.Quards(ID);

    int x = quards.x;
    int y = quards.y;

    if (ID < a.width * a.height)
    {
        ag.Add(x, y, apply(a.Get(x, y), -funcID) * bg.Get(x, y));
    }
}