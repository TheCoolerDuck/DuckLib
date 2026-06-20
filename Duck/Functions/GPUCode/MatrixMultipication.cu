
#include "Functions\\GPUCode\\GPU_Matrix.h"

extern "C" __global__ void Main(Matrix a, Matrix b, Matrix r)
{

    extern __shared__ float threadSums[];

    int com = a.height;

    int position = blockIdx.x * blockDim.x + threadIdx.x;
    int blockID = threadIdx.x * blockDim.y + threadIdx.y;

    int2 quards = r.Quards(position);

    int x = quards.x;
    int y = quards.y;

    int itemsPerThread = (com + blockDim.y - 1) / blockDim.y;

    if (position < r.width * r.height)
    {

        threadSums[blockID] = 0.0;

        int s = threadIdx.y * itemsPerThread;

        for (int j = 0; j < itemsPerThread; j++)
        {
            int i = j + s;
            if (i < com)
            {
                float aV = a.Get(x, i);
                float bV = b.Get(i, y);
                threadSums[blockID] += aV * bV;
            }
        }
    }
    for (int stride = blockDim.y / 2 + 1; stride > 0; stride >>= 1)
    {
        __syncthreads();
        if (threadIdx.y < stride && position < r.width * r.height && threadIdx.y + stride < com)
            threadSums[blockID] += threadSums[blockID + stride];
    }
    if (threadIdx.y == 0 && position < r.width * r.height)
        r.Add(x, y, threadSums[blockID]);

}