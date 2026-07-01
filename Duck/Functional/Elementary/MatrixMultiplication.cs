using Duck.Functional.Parameters;
using Duck.Functions.Parameters;
using Duck.Management;
using Duck.Matrix_Utilities;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System.Drawing;

namespace Duck.Functional.Elementary
{
    internal class MatrixMultiplication : IBasicFunction<DoubleMatrix>
    {
        protected override Matrix ApplyCPU(DoubleMatrix p)
        {
            int commonAxis = p.a.shape.height;

            float[,] values = new float[p.a.shape.width, p.b.shape.height];
            MatrixCPU a = (MatrixCPU)p.a.matrixBase;
            MatrixCPU b = (MatrixCPU)p.b.matrixBase;

            CPUManager.RunTask(0, p.a.shape.width, 0, p.b.shape.height, (x, y) =>
            {
                float sum = 0;
                for (int i = 0; i < commonAxis; i++)
                {
                    float aV = a[x, i, p.a.transposed];
                    float bV = b[i, y, p.b.transposed];
                    sum += aV * bV;
                }
                values[x, y] = sum;
            });

            p.result = new Matrix(values, new MatrixOptions() { Device = Device.CPU, HasGrad = ((IParameter)p).ResultHasGradient() }, new BackwardContext<DoubleMatrix>(this, p));

            return p.result;
        }

        protected override void ApplyGradientCPU(DoubleMatrix p)
        {
            MatrixCPU a = (MatrixCPU)p.a.matrixBase;
            MatrixCPU b = (MatrixCPU)p.b.matrixBase;
            MatrixCPU r = (MatrixCPU)p.result!.matrixBase;

            int m = p.a.shape.width;   
            int n = p.b.shape.height;  
            int k = p.a.shape.height; 

            if (a.hasGradient)
            {
                CPUManager.RunTask(0, m, 0, k, (x, i) =>
                {
                    float grad = 0;

                    for (int y = 0; y < n; y++)
                    {
                        float dC = r.GetGradient(x, y, p.result.transposed);
                        float bVal = b[i, y, p.b.transposed];
                        grad += dC * bVal;
                    }

                    a.AddGradient(x, i, grad, p.a.transposed);
                });
            }

            if (b.hasGradient)
            {
                CPUManager.RunTask(0, k, 0, n, (i, y) =>
                {
                    float grad = 0;

                    for (int x = 0; x < m; x++)
                    {
                        float dC = r.GetGradient(x, y, p.result.transposed);
                        float aVal = a[x, i, p.a.transposed];
                        grad += dC * aVal;
                    }

                    b.AddGradient(i, y, grad, p.b.transposed);
                });
            }
        }

        protected override Matrix ApplyGPU(DoubleMatrix p)
        {
            int size = p.b.shape.height * p.a.shape.width;
            int commonAxis = p.a.shape.height;

            CudaDeviceVariable<float> values = new(size);
            p.result = new Matrix((p.a.shape.width, p.b.shape.height), values, new BackwardContext<DoubleMatrix>(this, p), ((IParameter)p).ResultHasGradient());

            int blockHeight = Math.Min((GPUManager.threads + size - 1) / size, commonAxis);

            applyKernel.BlockDimensions = new dim3(64, blockHeight);
            applyKernel.GridDimensions = (size + 63) / 64;
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

        protected override void ValidateParameters(DoubleMatrix p)
        {
            if (p.a.shape.height != p.b.shape.width)
                throw new ArgumentException($"Matrices of incorrect shape: A: {p.a.shape}, B: {p.b.shape}");
        }

        private static CudaKernel? _applyKernel;
        public static CudaKernel applyKernel => _applyKernel ??= GPUManager.Compile(File.ReadAllText("Functional\\GPUCode\\MatrixMultipication.cu"));

    }
}