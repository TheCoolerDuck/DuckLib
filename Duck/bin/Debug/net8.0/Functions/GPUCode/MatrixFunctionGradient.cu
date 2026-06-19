
#include "Functions\\GPUCode\\GPU_Matrix.h"
#include "Functions\\GPUCode\\MatrixFunctionsApplyGradient.h"

//a must be the same size as result
extern "C" __global__ void Main(Matrix a, Matrix b, Matrix ag, Matrix bg, Matrix rg, int funcID)
{
    extern __shared__ float threadSums[];

    int ID = blockIdx.x * blockDim.x + threadIdx.x;

    int largeSize = a.width * a.height;
    int smallSize = b.width * b.height;

    bool bIsSame = b.width == a.width;
    bool bIsVector = b.width == 1;
    bool bIsScalar = smallSize == 1;

    int collisionSize = (largeSize + smallSize - 1) / smallSize;
    int itemsPerThread = (collisionSize + blockDim.y - 1) / blockDim.y;

    int x = groupID / maxHeight;
    int y = groupID % maxHeight;

    if (groupID < a.width * a.height)
    {
        int ax = a.width == 1 ? 0 : x;
        int ay = a.height == 1 ? 0 : y;

        int bx = b.width == 1 ? 0 : x;
        int by = b.height == 1 ? 0 : y;

        float2 grads = apply(a.get(ax, ay), b.get(bx, by));

        float grad = rg.get(x, y);

        ag.Add(x, y, grads.x * grad);

        if (bIsSame)
            bg.Add(x, y, grads.y * grad);
        else
            threadSums[blockID] = grads.y * grad;
    }

    for (int stride = b.width / 2 + 1; stride > 0; stride /= 2)
    {
        {
            __syncthreads();
            if (y < stride && ID < a.width * a.height && y + stride < b.width)
                threadSums[blockID] += threadSums[blockID + stride];
        }
    }
    if (i == 0 && groupID < a.width * a.height)
        bg.Add(x, y, threadSums[blockID]);
}