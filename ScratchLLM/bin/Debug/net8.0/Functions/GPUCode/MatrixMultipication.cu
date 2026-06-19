
#include "Functions\\GPUCode\\GPU_Matrix.h"

extern "C" __global__ void Main(Matrix a, Matrix b, Matrix c)
{

    extern __shared__ float threadSums[];
    int com = a.width;
    int pID = blockIdx.x * blockDim.x + threadIdx.x;
    int ID = threadIdx.x * blockDim.y + threadIdx.y;
    int x = pID / c.height;
    int y = pID % c.height;
    int i = threadIdx.y;
    int itemsPerThread = (com + blockDim.y - 1) / blockDim.y;
    if (pID < c.width * c.height)
    {
        {

            threadSums[ID] = 0.0;
            int s = i * itemsPerThread;
            for (int j = 0; j < itemsPerThread; j++)
            {
                {
                    int k = j + s;
                    if (k < com)
                    {
                        {
                            float aV = a.Get(y, k);
                            float bV = b.Get(k, x);
                            threadSums[ID] += aV * bV;
                        }
                    }
                }
            }
        }
    }
    for (int stride = blockDim.y / 2 + 1; stride > 0; stride /= 2)
    {
        {
            __syncthreads();
            if (i < stride && pID < c.width * c.height && i + stride < com)
                threadSums[ID] += threadSums[ID + stride];
        }
    }
    if (i == 0 && pID < c.width * c.height)
        c.Add(x, y, threadSums[ID]);

}