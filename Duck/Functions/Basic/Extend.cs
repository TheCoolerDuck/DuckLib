using Duck.Functions.Parameters;
using Duck.Management;
using Duck.Matrix_Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Basic
{
    public class Extend(FunctionType type) : IBasicFunction<MatrixAndIndex>
    {
        public FunctionType type = Validate(type);

        private static FunctionType Validate(FunctionType t)
        {
            if (t == FunctionType.Whole)
                throw new ArgumentException("FunctionType.Whole is not allowed.", nameof(t));
            return t;
        }


        public static Matrix ApplyWhole(Matrix m, int width, int height)
        {
            Matrix result;

            if (width > height)
            {
                m = new Extend(FunctionType.Column).Apply((m, height));
                result = new Extend(FunctionType.Row).Apply((m, width));
            }
            else
            {
                m = new Extend(FunctionType.Row).Apply((m, width));
                result = new Extend(FunctionType.Column).Apply((m, height));
            }

            return result;
        }

        protected override Matrix ApplyCPU(MatrixAndIndex p)
        {
            if (type == FunctionType.Row && !p.m.IsColVector() && !p.m.IsScalar())
                throw new ArgumentException($"matrix must be a column vector for row wise extension {p.m}");

            if (type == FunctionType.Column && !p.m.IsRowVector() && !p.m.IsScalar())
                throw new ArgumentException($"matrix must be a row vector for column wise extension {p.m}");

            MatrixCPU m = (MatrixCPU)p.m.matrixBase;

            int width = type == FunctionType.Row ? p.i : m.shape.width;
            int height = type == FunctionType.Column ? p.i : m.shape.height;

            float[,] values = new float[width, height];

            CPUManager.RunTask(0, width, 0, height, (x, y) =>
            {
                int mx = m.shape.width == 1 ? 0 : x;
                int my = m.shape.height == 1 ? 0 : y;

                values[x, y] = m.values[mx, my];
            });

            p.result = new Matrix(values, new MatrixOptions() { Device = Device.CPU }, new BackwardContext<MatrixAndIndex>(this, p));
            return p.result;
        }

        protected override Matrix ApplyGPU(MatrixAndIndex p)
        {
            throw new NotImplementedException();
        }

        protected override void ApplyGradientCPU(MatrixAndIndex p)
        {
            MatrixCPU m = (MatrixCPU)p.m.matrixBase;
            MatrixCPU r = (MatrixCPU)p.result!.matrixBase;

            CPUManager.RunTask(0, r.shape.width, 0, r.shape.height, (x, y) =>
            {
                int mx = m.shape.width == 1 ? 0 : x;
                int my = m.shape.height == 1 ? 0 : y;

                m.AddGradient(mx, my, r.Get(x, y, p.result.transposed), p.m.transposed);
            });
        }

        protected override void ApplyGradientGPU(MatrixAndIndex p)
        {
            throw new NotImplementedException();
        }
    }
}
