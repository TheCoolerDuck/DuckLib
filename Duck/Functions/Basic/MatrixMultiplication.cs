using Duck.CustomLLM.Library.Objects.MatrixObjects;
using Duck.Functions.Parameters;
using Duck.Management;
using Duck.Matrix_Utilities;

namespace Duck.Functions.Basic
{
    internal class MatrixMultiplication : IBasicFunction<DoubleMatrix>
    {
        public Matrix Apply(DoubleMatrix p)
        {
            // A is (aW x aH), B is (bW x bH)
            // For matmul: A's width must equal B's height
            if (p.a.shape.width != p.b.shape.height)
                throw new ArgumentException($"Matrices of incorrect shape: A: {p.a.shape}, B: {p.b.shape}");
            if (p.a.device != p.b.device)
                throw new ArgumentException("Matrices must be on the same device");

            if (p.a.device == Device_Management.Device.CPU)
            {
                // Output is (bW x aH) — width from B, height from A
                double[,] values = new double[p.b.shape.width, p.a.shape.height];
                MatrixCPU a = (MatrixCPU)p.a.matrixBase;
                MatrixCPU b = (MatrixCPU)p.b.matrixBase;

                // x iterates output width (0..bW), y iterates output height (0..aH)
                CPUThreadManager.RunTask(0, p.b.shape.width, 0, p.a.shape.height, (x, y) =>
                {
                    double sum = 0;
                    for (int i = 0; i < p.a.shape.width; i++)
                        sum += a[i, y, p.a.transposed] * b[x, i, p.b.transposed];
                    values[x, y] = sum;
                });

                p.result = new Matrix(values, new BackwardContext<DoubleMatrix>(this, p), Device_Management.Device.CPU);
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
                MatrixCPU r = (MatrixCPU)p.result.matrixBase;

                // dL/dA[i, y] = sum_x( dL/dR[x, y] * B[x, i] )
                // x in 0..bW (result width), y in 0..aH (result height), i in 0..aW
                if (a.hasGradient)
                {
                    CPUThreadManager.RunTask(0, p.a.shape.width, 0, p.a.shape.height, (i, y) =>
                    {
                        double grad = 0;
                        for (int x = 0; x < p.b.shape.width; x++)
                            grad += r.GetGradient(x, y, p.result.transposed) * b[x, i, p.b.transposed];
                        a.AddGradient(i, y, grad, p.a.transposed);
                    });
                }

                // dL/dB[x, i] = sum_y( dL/dR[x, y] * A[i, y] )
                // x in 0..bW (result width), i in 0..bH (= aW), y in 0..aH
                if (b.hasGradient)
                {
                    CPUThreadManager.RunTask(0, p.b.shape.width, 0, p.b.shape.height, (x, i) =>
                    {
                        double grad = 0;
                        for (int y = 0; y < p.a.shape.height; y++)
                            grad += r.GetGradient(x, y, p.result.transposed) * a[i, y, p.a.transposed];
                        b.AddGradient(x, i, grad, p.b.transposed);
                    });
                }
            }
            else
            {
                throw new NotImplementedException();
            }
        }
    }
}