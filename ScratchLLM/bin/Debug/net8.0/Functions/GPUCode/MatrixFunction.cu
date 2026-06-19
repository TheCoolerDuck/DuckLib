
#include "Functions\\GPUCode\\GPU_Matrix.h"
#include "Functions\\GPUCode\\MatrixFunctionsApply.h"

extern "C" __global__ void Main(Matrix a, Matrix b, Matrix c, int funcID)
{
    int ID = blockIdx.x * blockDim.x + threadIdx.x;
    int x = ID % c.height;
    int y = ID / c.height;
    if (ID < c.width * c.height)
    {
        int ax = a.width == 1 ? 0 : x;
        int ay = a.height == 1 ? 0 : y;

        int bx = b.width == 1 ? 0 : x;
        int by = b.height == 1 ? 0 : y;

        float aV = a.Get(ax, ay);
        float bV = b.Get(bx, by);
        float rV = apply(aV, bV, funcID);
        c.Add(x, y, rV);
    }
}