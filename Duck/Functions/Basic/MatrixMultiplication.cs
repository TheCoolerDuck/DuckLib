using Duck.Functions.Parameters;
using Duck.Management;
using Duck.Matrix_Utilities;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System.Drawing;

namespace Duck.Functions.Basic
{
    internal class MatrixMultiplication : IBasicFunction<DoubleMatrix>
    {

        private static CudaKernel? _applyKernel;
        public static CudaKernel applyKernel => _applyKernel ??= GPUManager.Compile(File.ReadAllText("Functions\\GPUCode\\MatrixMultipication.cu"));

        protected override Matrix ApplyCPU(DoubleMatrix p)
        {
            if (p.a.shape.width != p.b.shape.height)
                throw new ArgumentException($"Matrices of incorrect shape: A: {p.a.shape}, B: {p.b.shape}");

            int commonAxis = p.a.shape.width;

            float[,] values = new float[p.b.shape.width, p.a.shape.height];
            MatrixCPU a = (MatrixCPU)p.a.matrixBase;
            MatrixCPU b = (MatrixCPU)p.b.matrixBase;

            CPUManager.RunTask(0, p.b.shape.width, 0, p.a.shape.height, (x, y) =>
            {
                float sum = 0;
                for (int i = 0; i < commonAxis; i++)
                    sum += a[i, y, p.a.transposed] * b[x, i, p.b.transposed];
                values[x, y] = sum;
            });

            p.result = new Matrix(values, new MatrixOptions() { Device = Device.CPU }, new BackwardContext<DoubleMatrix>(this, p));

            return p.result;
        }

        protected override void ApplyGradientCPU(DoubleMatrix p)
        {
            MatrixCPU a = (MatrixCPU)p.a.matrixBase;
            MatrixCPU b = (MatrixCPU)p.b.matrixBase;
            MatrixCPU r = (MatrixCPU)p.result!.matrixBase;

            if (a.hasGradient)
            {
                CPUManager.RunTask(0, p.a.shape.width, 0, p.a.shape.height, (x, y) =>
                {
                    float grad = 0;
                    for (int i = 0; i < p.b.shape.width; i++)
                        grad += r.GetGradient(i, y, p.result.transposed) * b[i, x, p.b.transposed];
                    a.AddGradient(x, y, grad, p.a.transposed);
                });
            }

            if (b.hasGradient)
            {
                CPUManager.RunTask(0, p.b.shape.width, 0, p.b.shape.height, (x, y) =>
                {
                    float grad = 0;
                    for (int i = 0; i < p.a.shape.height; i++)
                        grad += r.GetGradient(x, i, p.result.transposed) * a[y, i, p.a.transposed];
                    b.AddGradient(x, y, grad, p.b.transposed);
                });
            }
        }

        protected override Matrix ApplyGPU(DoubleMatrix p)
        {
            if (p.a.shape.width != p.b.shape.height)
                throw new ArgumentException($"Matrices of incorrect shape: A: {p.a.shape}, B: {p.b.shape}");

            int size = p.b.shape.width * p.a.shape.height;
            int commonAxis = p.a.shape.width;

            CudaDeviceVariable<float> values = new(size);
            p.result = new Matrix((p.b.shape.width, p.a.shape.height), values, new BackwardContext<DoubleMatrix>(this, p));

            int blockHeight = Math.Min((GPUManager.threads + size - 1) / size, commonAxis);

            applyKernel.BlockDimensions = new dim3(64, blockHeight);
            applyKernel.GridDimensions = (64 + size - 1) / 64;
            applyKernel.DynamicSharedMemory = (uint)(64 * blockHeight * sizeof(float));

            applyKernel.Run(p.a.GPUValues(), p.b.GPUValues(), p.result.GPUValues());

            return p.result;
        }

        protected override void ApplyGradientGPU(DoubleMatrix p)
        {
            if (p.a.matrixBase.hasGradient)
            {
                int size = p.a.shape.width * p.a.shape.height;
                int commonAxis = p.result!.shape.width;

                int blockHeight = Math.Min((GPUManager.threads + size - 1) / size, commonAxis);

                applyKernel.BlockDimensions = new dim3(64, blockHeight);
                applyKernel.GridDimensions = (64 + size - 1) / 64;
                applyKernel.DynamicSharedMemory = (uint)(64 * blockHeight * sizeof(float));

                applyKernel.Run(p.result.GPUGradient(), p.b.T().GPUValues(), p.a.GPUGradient());
            }

            if (p.b.matrixBase.hasGradient)
            {
                int size = p.b.shape.width * p.b.shape.height;
                int commonAxis = p.a.shape.height;

                int blockHeight = Math.Min((GPUManager.threads + size - 1) / size, commonAxis);

                applyKernel.BlockDimensions = new dim3(64, blockHeight);
                applyKernel.GridDimensions = (64 + size - 1) / 64;
                applyKernel.DynamicSharedMemory = (uint)(64 * blockHeight * sizeof(float));

                applyKernel.Run(p.a.T().GPUValues(), p.result!.GPUGradient(), p.b.GPUGradient());
            }
        }
    }
}