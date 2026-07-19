
#include "Functional\\GPUCode\\GPU_Matrix.h"

extern "C" __global__ void Main(bool isCol, Matrix m, Matrix r, int* indices)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < r.width * r.height)
    {
        int vectorLen = isCol ? m.width : m.height;

        int srcRow = isCol ? i % vectorLen : indices[i / vectorLen];
        int srcCol = isCol ? indices[i / vectorLen] : i % vectorLen;
        int dstRow = isCol ? i / vectorLen : i % vectorLen;
        int dstCol = isCol ? i % vectorLen : i / vectorLen;



        r.Add(dstRow, dstCol, m.Get(srcRow, srcCol));
    }
}