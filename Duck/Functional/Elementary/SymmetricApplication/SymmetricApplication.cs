using Duck.Functional.Parameters;
using Duck.Functions.Value.Double;
using Duck.Functions.Value.Single;
using Duck.Management;
using Duck.Matrix_Utilities;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functional.Elementary.SymmetricApplication
{
    public class SymmetricApplication<T> : BasicFunction<SingleMatrix> where T : IDoubleValueFunction
    {
        protected override void ApplyCPU(SingleMatrix p, float[,] result)
        {
            float[] values = new float[p.m.size];
            float[][] cache = new float[(int)MathF.Ceiling(MathF.Log2(values.Length))][];

            CPUManager.RunTask(0, p.m.size, i =>
            {
                (int x, int y) = p.m.XYFromIndex(i);

                values[i] = ((MatrixCPU)p.m.matrixBase)[x, y, p.m.transposed];
            });

            int j = 0;
            while (values.Length > 1)
            {
                int half = (values.Length + 1) / 2;
                float[] newValues = new float[half];
                CPUManager.RunTask(0, values.Length / 2, i =>
                {
                    newValues[i] = T.Apply(values[i], values[i + half]);
                });

                if (values.Length % 2 == 1)
                    newValues[half - 1] = values[half - 1];

                cache[j++] = values;
                values = newValues;
            }

            p.cache = cache;

            result[0, 0] = values[0];
        }

        protected override void ApplyGPU(SingleMatrix p, Matrix result)
        {
            int cacheSize = (int)MathF.Ceiling(MathF.Log2(p.m.size));

            CudaDeviceVariable<CUdeviceptr> cache = new(cacheSize);
            CUdeviceptr[] cacheArrays = new CUdeviceptr[cacheSize];

            int size = p.m.size;

            for (int i = 0; i < cacheSize; i++)
            {
                cacheArrays[i] = new CudaDeviceVariable<float>(size).DevicePointer;

                size = (size + 1) / 2;
            }

            cache.CopyToDevice(cacheArrays);

            GPUManager.SetKernelSize(applyKernel, p.m.size);

            applyKernel.Run(p.m.GPUValues(), result.GPUValues(), cache.DevicePointer, GPUManager.StableHash(typeof(T).Name));
        }

        protected override void ApplyGradientCPU(SingleMatrix p)
        {
            float[][] cache = (float[][])p.cache!;
            float[] grad = new float[p.m.size];

            Array.Fill(grad, 1);

            for (int j = cache.Length - 1; j >= 0; j--)
            {
                float[] values = cache[j];
                int half = (values.Length + 1) / 2;

                CPUManager.RunTask(0, values.Length / 2, i =>
                {
                    (float ag, float bg) = T.ApplyDerivative(values[i], values[i + half]);
                    float g = grad[i];
                    grad[i] = ag * g;
                    grad[i + half] = bg * g;
                });
            }

            float rg = ((MatrixCPU)p.result!.matrixBase).GetGradient(0, 0, p.result.transposed);

            CPUManager.RunTask(0, p.m.size, i =>
            {
                (int x, int y) = p.m.XYFromIndex(i);

                ((MatrixCPU)p.m.matrixBase).AddGradient(x, y, grad[i] * rg, p.m.transposed);
            });
        }

        protected override void ApplyGradientGPU(SingleMatrix p)
        {
            throw new NotImplementedException();
        }

        protected override void ValidateParameters(SingleMatrix p) { }

        protected override (int width, int height) GetResultShape(SingleMatrix p)
        {
            return (1, 1);
        }

        private readonly static CudaKernel applyKernel = GPUManager.Compile("float x, float y", "float", "nanf(\"\")", typeof(IDoubleValueFunction));
    }
}
