using Duck.Functional.Parameters;
using Duck.Functions.Parameters;
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

namespace Duck.Functional.Elementary
{
    public class SymmetricApplication<T> : IBasicFunction<SingleMatrix> where T : IDoubleValueFunction
    {
        protected override Matrix ApplyCPU(SingleMatrix p)
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

            p.result = new Matrix(new float[,] { { values[0] } }, new MatrixOptions() { Device = Device.CPU, HasGrad = ((IParameter)p).ResultHasGradient() }, new BackwardContext<SingleMatrix>(this, p));
            return p.result;
        }

        protected override Matrix ApplyGPU(SingleMatrix p)
        {
            CudaDeviceVariable<float> result = new(1);
            p.result = new Matrix((1, 1), result, new BackwardContext<SingleMatrix>(this, p), ((IParameter)p).ResultHasGradient());

            int cacheSize = (int)MathF.Ceiling(MathF.Log2(p.m.size));

            CudaDeviceVariable<CUdeviceptr> cache = new(cacheSize);
            CUdeviceptr[] cacheArrays = [.. Enumerable.Range(0, cacheSize).Select(i => new CudaDeviceVariable<float>((int)MathF.Ceiling(MathF.Pow(p.m.size, -2 * i))).DevicePointer)];

            cache.CopyToDevice(cacheArrays);

            applyKernel.BlockDimensions = 64;
            applyKernel.GridDimensions = (p.m.size + 63) / 64;

            applyKernel.Run(p.m.GPUValues(), p.result.GPUValues(), cache.DevicePointer, GPUManager.StableHash(typeof(T).Name));

            return p.result;
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

        private static CudaKernel? _applyKernel;
        public static CudaKernel applyKernel => _applyKernel ??= GPUManager.Compile(File.ReadAllText("Functional\\GPUCode\\SymmetricApplication.cu"));
    }
}
