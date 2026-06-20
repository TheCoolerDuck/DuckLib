
#include "Functions\\GPUCode\\GPU_Matrix.h"
#include "Functions\\GPUCode\\MatrixFunctionsApplyGradient.h"

//a must be the same size as result
extern "C" __global__ void Main(Matrix a, Matrix b, Matrix ag, Matrix bg, Matrix rg, int funcID)
{
    int ID = blockIdx.x * blockDim.x + threadIdx.x;

    int2 quards = r.Quards(ID);

    int x = quards.x;
    int y = quards.y;

    if (ID < a.width * a.height)
    {
        float2 grads = apply(a.get(x, y), b.get(x, y));

        float grad = rg.get(x, y);

        ag.Add(x, y, grads.x * grad);
        bg.Add(x, y, grads.y * grad);
    }
}