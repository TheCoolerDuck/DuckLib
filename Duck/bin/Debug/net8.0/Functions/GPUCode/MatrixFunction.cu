
#include "Functions\\GPUCode\\GPU_Matrix.h"
#include "Functions\\GPUCode\\MatrixFunctionsApply.h"

extern "C" __global__ void Main(Matrix a, Matrix b, Matrix r, int funcID)
{
    int ID = blockIdx.x * blockDim.x + threadIdx.x;

    int2 quards = r.Quards(ID);

    int x = quards.x;
    int y = quards.y;

    if (ID < r.width * r.height)
    {
        int ax = a.width == 1 ? 0 : x;
        int ay = a.height == 1 ? 0 : y;

        int bx = b.width == 1 ? 0 : x;
        int by = b.height == 1 ? 0 : y;

        float aV = a.Get(ax, ay);
        float bV = b.Get(bx, by);
        float rV = apply(aV, bV, funcID);
        r.Add(x, y, rV);
    }
}