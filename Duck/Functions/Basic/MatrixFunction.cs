using Duck.CustomLLM.Library.Objects.MatrixObjects;
using Duck.Functions.Parameters;
using Duck.Functions.Value.Double;
using Duck.Management;
using Duck.Matrix_Utilities;
using ManagedCuda;
using System;

namespace Duck.Functions.Basic
{
    internal class MatrixFunction<T> : IBasicFunction<DoubleMatrix> where T : IDoubleValueFunction
    {
        public Matrix Apply(DoubleMatrix p)
        {
            if (p.a.device != p.b.device)
                throw new ArgumentException("Matrices must be on the same device");

            bool sameSize = p.a.shape == p.b.shape;
            bool hasScalar = p.a.IsScalar() || p.b.IsScalar();
            bool isRowVector = p.a.shape.width == p.b.shape.width
                               && (p.a.IsRowVector() || p.b.IsRowVector());
            bool isColVector = p.a.shape.height == p.b.shape.height
                               && (p.a.IsColVector() || p.b.IsColVector());

            if (!(sameSize || hasScalar || isRowVector || isColVector))
                throw new ArgumentException(
                    $"Matrices of incompatible shape: A: {p.a.shape}, B: {p.b.shape}");

            int outWidth = Math.Max(p.a.shape.width, p.b.shape.width);
            int outHeight = Math.Max(p.a.shape.height, p.b.shape.height);

            if (p.a.device == Device_Management.Device.CPU)
            {
                float[,] values = new float[outWidth, outHeight];

                CPUManager.RunTask(0, outWidth, 0, outHeight, (x, y) =>
                {
                    var (ax, ay, bx, by) = GetBroadcastCoords(p.a, p.b, x, y);
                    float aVal = ((MatrixCPU)p.a.matrixBase)[ax, ay, p.a.transposed];
                    float bVal = ((MatrixCPU)p.b.matrixBase)[bx, by, p.b.transposed];
                    values[x, y] = T.Apply(aVal, bVal);
                });

                p.result = new Matrix(
                    values,
                    new BackwardContext<DoubleMatrix>(this, p),
                    Device_Management.Device.CPU);
            }
            else
            {
                throw new NotImplementedException();
            }

            return p.result;
        }

        public void ApplyGradient(DoubleMatrix p)
        {
            if (p.result == null)
                throw new ArgumentException("Params must have a result");

            if (p.a.device == Device_Management.Device.CPU)
            {
                MatrixCPU a = (MatrixCPU)p.a.matrixBase;
                MatrixCPU b = (MatrixCPU)p.b.matrixBase;
                MatrixCPU result = (MatrixCPU)p.result.matrixBase;

                // Iterate over the OUTPUT shape, not p.a's shape, to cover all grad contributions
                int outWidth = p.result.shape.width;
                int outHeight = p.result.shape.height;

                CPUManager.RunTask(0, outWidth, 0, outHeight, (x, y) =>
                {
                    var (ax, ay, bx, by) = GetBroadcastCoords(p.a, p.b, x, y);

                    float aVal = a[ax, ay, p.a.transposed];
                    float bVal = b[bx, by, p.b.transposed];
                    float rGrad = result.GetGradient(x, y, p.result.transposed);

                    (float aGrad, float bGrad) = T.ApplyDerivative(aVal, bVal);

                    a.AddGradient(ax, ay, aGrad * rGrad, p.a.transposed);
                    b.AddGradient(bx, by, bGrad * rGrad, p.b.transposed);
                });
            }
            else
            {
                throw new NotImplementedException();
            }
        }

        // --- Broadcasting helpers (mirrors getMatQuards in the reference) ---

        private static (int ax, int ay, int bx, int by) GetBroadcastCoords(
            Matrix a, Matrix b, int x, int y)
        {
            int ax = a.IsScalar() || a.IsColVector() ? 0 : x;
            int ay = a.IsScalar() || a.IsRowVector() ? 0 : y;
            int bx = b.IsScalar() || b.IsColVector() ? 0 : x;
            int by = b.IsScalar() || b.IsRowVector() ? 0 : y;
            return (ax, ay, bx, by);
        }

        private static CudaKernel? _applyKernel;
        private static CudaKernel? _gradientKernel;
        public static CudaKernel applyKernel => _applyKernel ??= GPUManager.Compile(applyCode);
        public static CudaKernel gradientKernel => _gradientKernel ??= GPUManager.Compile(gradientCode);


        private const string applyCode = $@"
            __device__ float getVal(float* m, int row, int col, int numRows, int numCols, bool transposed)
            {{
                if (transposed)
                    return m[row * numCols + col];
                else
                    return m[col * numRows + row];
            }}
            extern ""C"" __global__ void Main(float* a, float* b, float* c,
                                                bool aT, bool bT,
                                                int w, int l, int h)
            {{
                extern __shared__ float threadSums[];

                int pID = blockIdx.x * blockDim.x + threadIdx.x;
                int ID  = threadIdx.x * blockDim.y + threadIdx.y; 
                int x   = pID / l;
                int y   = pID % l;
                int i   = threadIdx.y;
                int itemsPerThread = (h + blockDim.y - 1) / blockDim.y;
                if (pID < w * l)
                {{

                    threadSums[ID] = 0.0;
                    int s = i * itemsPerThread;
                    for (int j = 0; j < itemsPerThread; j++)
                    {{
                        int k = j + s;
                        if (k < h)
                        {{
                            float aV = getVal(a, k, y, l, h, aT); 
                            float bV = getVal(b, x, k, w, l, bT);
                            threadSums[ID] += aV * bV;
                        }}
                    }}
                }}
                for (int stride = blockDim.y / 2 + 1; stride > 0; stride /= 2)
                {{
                    __syncthreads();
                    if (i < stride && pID < w * l && i + stride < l)
                        threadSums[ID] += threadSums[ID + stride];
                }}
                if (i == 0 && pID < w * l)
                    c[y * w + x] = threadSums[ID];
            }}";

        private const string gradientCode = $@"
            __device__ float getVal(float* m, int row, int col, int numRows, int numCols, bool transposed)
            {{
                if (transposed)
                    return m[row * numCols + col];
                else
                    return m[col * numRows + row];
            }}
            extern ""C"" __global__ void Main(float* outGrad, float* o, float* inGrad, //swap a and be for b's gradient
                                                bool thisT, bool otherT, bool rT, //rT is restult transposed and is true for b's gradient
                                                int w, int l, int h)
            {{
                extern __shared__ float threadSums[];

                int pID = blockIdx.x * blockDim.x + threadIdx.x;
                int ID  = threadIdx.x * blockDim.y + threadIdx.y; 
                int x   = pID / l;
                int y   = pID % l;
                int i   = threadIdx.y;
                int itemsPerThread = (h + blockDim.y - 1) / blockDim.y;
                if (pID < w * l)
                {{

                    threadSums[ID] = 0.0;
                    int s = i * itemsPerThread;
                    for (int j = 0; j < itemsPerThread; j++)
                    {{
                        int k = j + s;
                        if (k < h)
                        {{
                            float bV    = getVal(o,      rT ? x : k, rT ? k : y, l, h, otherT); 
                            float gradV = getVal(inGrad, rT ? y : k, rT ? k : x, w, l, false);
                            threadSums[ID] += gradV * bV;
                        }}
                    }}
                }}
                for (int stride = blockDim.y / 2 + 1; stride > 0; stride /= 2)
                {{
                    __syncthreads();
                    if (i < stride && pID < w * l && i + stride < l)
                        threadSums[ID] += threadSums[ID + stride];
                }}
                if (i == 0 && pID < w * l)
                    outGrad[thisT ? y * w + x : x * h + y] = threadSums[ID];
            }}";
    }
}