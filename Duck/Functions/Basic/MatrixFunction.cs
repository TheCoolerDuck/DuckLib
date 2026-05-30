using Duck.CustomLLM.Library.Objects.MatrixObjects;
using Duck.Functions.Parameters;
using Duck.Functions.Value.Double;
using Duck.Management;
using Duck.Matrix_Utilities;
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
            bool hasScalar = IsScalar(p.a) || IsScalar(p.b);
            bool isRowVector = p.a.shape.width == p.b.shape.width
                               && (IsRowVector(p.a) || IsRowVector(p.b));
            bool isColVector = p.a.shape.height == p.b.shape.height
                               && (IsColVector(p.a) || IsColVector(p.b));

            if (!(sameSize || hasScalar || isRowVector || isColVector))
                throw new ArgumentException(
                    $"Matrices of incompatible shape: A: {p.a.shape}, B: {p.b.shape}");

            int outWidth = Math.Max(p.a.shape.width, p.b.shape.width);
            int outHeight = Math.Max(p.a.shape.height, p.b.shape.height);

            if (p.a.device == Device_Management.Device.CPU)
            {
                double[,] values = new double[outWidth, outHeight];

                CPUThreadManager.RunTask(0, outWidth, 0, outHeight, (x, y) =>
                {
                    var (ax, ay, bx, by) = GetBroadcastCoords(p.a, p.b, x, y);
                    double aVal = ((MatrixCPU)p.a.matrixBase)[ax, ay, p.a.transposed];
                    double bVal = ((MatrixCPU)p.b.matrixBase)[bx, by, p.b.transposed];
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

                CPUThreadManager.RunTask(0, outWidth, 0, outHeight, (x, y) =>
                {
                    var (ax, ay, bx, by) = GetBroadcastCoords(p.a, p.b, x, y);

                    double aVal = a[ax, ay, p.a.transposed];
                    double bVal = b[bx, by, p.b.transposed];
                    double rGrad = result.GetGradient(x, y, p.result.transposed);

                    (double aGrad, double bGrad) = T.ApplyDerivative(aVal, bVal);

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
            int ax = IsScalar(a) || IsColVector(a) ? 0 : x;
            int ay = IsScalar(a) || IsRowVector(a) ? 0 : y;
            int bx = IsScalar(b) || IsColVector(b) ? 0 : x;
            int by = IsScalar(b) || IsRowVector(b) ? 0 : y;
            return (ax, ay, bx, by);
        }

        private static bool IsScalar(Matrix m) => m.shape.width == 1 && m.shape.height == 1;
        private static bool IsRowVector(Matrix m) => m.shape.height == 1;
        private static bool IsColVector(Matrix m) => m.shape.width == 1;
    }
}