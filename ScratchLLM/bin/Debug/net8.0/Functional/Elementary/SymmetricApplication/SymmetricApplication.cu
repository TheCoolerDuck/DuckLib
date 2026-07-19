
#include "Functional\\GPUCode\\GPU_Matrix.h"

extern "C" __global__ void Main(Matrix m, Matrix r, float** cache, int funcID)
{
    int len = m.width * m.height;
    int j = 0;
    int i = threadIdx.x;
    int2 pos = m.Quards(i);
    int2 posPlusHalf = m.Quards(i + (len + 1) / 2);
    while (len > 1)
    {
        __syncthreads();
        int half = (len + 1) / 2;
        if (i < len / 2)
            cache[j][i] = j == 0 ? 
            apply(m.Get(pos.x, pos.y), m.Get(posPlusHalf.x, posPlusHalf.y), funcID) :
            apply(cache[j - 1][i], cache[j - 1][i + half], funcID);

        if (len % 2 == 1 && i == half - 1)
            cache[j][i] = j == 0 ? m.Get(pos.x, pos.y) : cache[j - 1][i];
        len = half;
        j++;
    }
    if (i == 0)
        r.Add(0, 0, cache[j-1][0]);

}