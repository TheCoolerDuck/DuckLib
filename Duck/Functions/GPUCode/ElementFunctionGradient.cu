
#include "Functions\\GPUCode\\GPU_Matrix.h"
#include "Functions\\GPUCode\\ElementFuntionsApply.h"


extern "C" __global__ void Main(Matrix a, Matrix ag, Matrix bg, int funcID)
{
    int ID = blockIdx.x * blockDim.x + threadIdx.x;
    int x = ID / a.height;
    int y = ID % a.height;
    if (ID < a.width * a.height)
    {
        ag.Add(x, y, apply(a.Get(x, y), -funcID) * bg.Get(x, y));
    }
}