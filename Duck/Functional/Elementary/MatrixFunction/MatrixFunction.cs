using Duck.Functional.Parameters;
using Duck.Functions.Value.Double;
using Duck.Functions.Value.Single;
using Duck.Management;
using Duck.Matrix_Utilities;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System;
using System.Drawing;
using System.Text;

namespace Duck.Functional.Elementary.MatrixFunction
{
    public class MatrixFunction<T> : BasicFunction<DoubleMatrix> where T : IDoubleValueFunction
    {
        protected override void ApplyCPU(DoubleMatrix p, float[,] result)
        { 
            MatrixCPU aCPU = (MatrixCPU)p.a.matrixBase;
            MatrixCPU bCPU = (MatrixCPU)p.b.matrixBase;

            CPUManager.RunTask(0, p.a.shape.width, 0, p.a.shape.height, (x, y) =>
            {
                result[x, y] = T.Apply(aCPU[x, y, p.a.transposed], bCPU[x, y, p.b.transposed]);
            });
        }

        protected override void ApplyGradientCPU(DoubleMatrix p)
        {
            MatrixCPU aCPU = (MatrixCPU)p.a.matrixBase;
            MatrixCPU bCPU = (MatrixCPU)p.b.matrixBase;
            MatrixCPU result = (MatrixCPU)p.result!.matrixBase;

            CPUManager.RunTask(0, p.result.shape.width, 0, p.result.shape.height, (x, y) =>
            {
                float aVal = aCPU[x, y, p.a.transposed];
                float bVal = bCPU[x, y, p.b.transposed];
                float rGrad = result.GetGradient(x, y, p.result.transposed);

                (float aGrad, float bGrad) = T.ApplyDerivative(aVal, bVal);

                aCPU.AddGradient(x, y, aGrad * rGrad, p.a.transposed);
                bCPU.AddGradient(x, y, bGrad * rGrad, p.b.transposed);
            });
        }

        protected override void ApplyGPU(DoubleMatrix p, Matrix result)
        {
            GPUManager.SetKernelSize(applyKernel, result.size);

            applyKernel.Run(p.a.GPUValues(), p.b.GPUValues(), result.GPUValues(), GPUManager.StableHash(typeof(T).Name));
        }

        protected override void ApplyGradientGPU(DoubleMatrix p)
        {
            GPUManager.SetKernelSize(applyKernel, p.result!.size);

            gradientKernel.Run(p.a.GPUValues(), p.b.GPUValues(), p.a.GPUGradient(), p.b.GPUGradient(), p.result.GPUGradient(), GPUManager.StableHash(typeof(T).Name));
        }

        private readonly static CudaKernel applyKernel = GPUManager.Compile("float x, float y", "float", "nanf(\"\")", typeof(IDoubleValueFunction));
        private readonly static CudaKernel gradientKernel = GPUManager.CompileGradient("float x, float y", "float2", "make_float2(nanf(\"\"), nanf(\"\"))", typeof(IDoubleValueFunction));

        protected override void ValidateParameters(DoubleMatrix p)
        {
            if (p.a.shape != p.b.shape)
                throw new ArgumentException($"Matrices must be of same shape {p.a} {p.b}");
        }

        protected override (int width, int height) GetResultShape(DoubleMatrix p)
        {
            return p.a.shape;
        }
    }
}