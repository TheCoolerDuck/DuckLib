using Duck.Functional.Parameters;
using Duck.Functions.Value.Single;
using Duck.Management;
using Duck.Matrix_Utilities;
using ManagedCuda;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functional.Elementary.ElementFunction
{
    public class ElementFunction<T> : BasicFunction<SingleMatrix> where T : ISingleValueFunction
    {
        protected override void ApplyCPU(SingleMatrix p, float[,] result)
        {
            CPUManager.RunTask(0, p.m.shape.width, 0, p.m.shape.height, (x, y) =>
            {
                result[x, y] = T.Apply(((MatrixCPU)p.m.matrixBase)[x, y, p.m.transposed]);
            });
        }

        protected override void ApplyGradientCPU(SingleMatrix p)
        {
            MatrixCPU m = (MatrixCPU)p.m.matrixBase;
            MatrixCPU result = (MatrixCPU)p.result!.matrixBase;

            CPUManager.RunTask(0, p.m.shape.width, 0, p.m.shape.height, (x, y) =>
            {
                float rGrad = result.GetGradient(x, y, p.result.transposed);
                float v = m[x, y, p.m.transposed];
                float mGrad = T.ApplyDerivative(v);
                m.AddGradient(x, y, mGrad * rGrad, p.m.transposed);
            });
        }
        protected override void ApplyGPU(SingleMatrix p, Matrix result)
        {
            GPUManager.SetKernelSize(gradientKernel, p.m.size);

            applyKernel.Run(p.m.GPUValues(), result.GPUValues(), GPUManager.StableHash(typeof(T).Name));
        }

        protected override void ApplyGradientGPU(SingleMatrix p)
        {
            GPUManager.SetKernelSize(gradientKernel, p.m.size);

            gradientKernel.Run(p.m.GPUValues(), p.m.GPUGradient(), p.result!.GPUGradient(), GPUManager.StableHash(typeof(T).Name));
        }

        private readonly static CudaKernel applyKernel = GPUManager.Compile("float x", "float", "nanf(\"\")", typeof(ISingleValueFunction));
        private readonly static CudaKernel gradientKernel = GPUManager.CompileGradient("float x", "float", "nanf(\"\")", typeof(ISingleValueFunction));

        protected override void ValidateParameters(SingleMatrix p) { }

        protected override (int width, int height) GetResultShape(SingleMatrix p)
        {
            return p.m.shape;
        }
    }
}
