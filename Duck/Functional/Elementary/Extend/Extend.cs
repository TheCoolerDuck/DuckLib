using Duck.Functional.Parameters;
using Duck.Management;
using Duck.Matrix_Utilities;
using ManagedCuda;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functional.Elementary.Extend
{
    public class Extend(FunctionType type) : BasicFunction<MatrixAndIndex>
    {
        public FunctionType type = Validate(type);

        private static FunctionType Validate(FunctionType t)
        {
            if (t == FunctionType.Whole)
                throw new ArgumentException("FunctionType.Whole is not allowed.", nameof(t));
            return t;
        }

        protected override void ApplyCPU(MatrixAndIndex p, float[,] result)
        {
            MatrixCPU m = (MatrixCPU)p.m.matrixBase;

            int width = type == FunctionType.Row ? p.i : p.m.shape.width;
            int height = type == FunctionType.Column ? p.i : p.m.shape.height;

            CPUManager.RunTask(0, width, 0, height, (x, y) =>
            {
                int mx = p.m.shape.width == 1 ? 0 : x;
                int my = p.m.shape.height == 1 ? 0 : y;

                result[x, y] = m[mx, my, p.m.transposed];
            });
        }

        protected override void ApplyGPU(MatrixAndIndex p, Matrix result)
        {
            GPUManager.SetKernelSize(applyKernel, result.size);

            applyKernel.Run(p.m.GPUValues(), result.GPUValues());
        }

        protected override void ApplyGradientCPU(MatrixAndIndex p)
        {
            MatrixCPU m = (MatrixCPU)p.m.matrixBase;
            MatrixCPU r = (MatrixCPU)p.result!.matrixBase;

            CPUManager.RunTask(0, r.shape.width, 0, r.shape.height, (x, y) =>
            {
                int mx = m.shape.width == 1 ? 0 : x;
                int my = m.shape.height == 1 ? 0 : y;

                m.AddGradient(mx, my, r.GetGradient(x, y, p.result.transposed), p.m.transposed);
            });
        }

        protected override void ApplyGradientGPU(MatrixAndIndex p)
        {
            throw new NotImplementedException();
        }

        protected override void ValidateParameters(MatrixAndIndex p)
        {
            if (type == FunctionType.Row && !p.m.IsColVector() && !p.m.IsScalar())
                throw new ArgumentException($"matrix must be a column vector for row wise extension {p.m}");

            if (type == FunctionType.Column && !p.m.IsRowVector() && !p.m.IsScalar())
                throw new ArgumentException($"matrix must be a row vector for column wise extension {p.m}");
        }

        protected override (int width, int height) GetResultShape(MatrixAndIndex p)
        {
            int width = type == FunctionType.Row ? p.i : p.m.shape.width;
            int height = type == FunctionType.Column ? p.i : p.m.shape.height;

            return (width, height); 
        }

        private readonly static CudaKernel applyKernel = GPUManager.Compile();

    }
}
