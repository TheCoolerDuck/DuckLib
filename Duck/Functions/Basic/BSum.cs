using Duck.Functions.Parameters;
using Duck.Management;
using Duck.Matrix_Utilities;

namespace Duck.Functions.Basic
{
    internal class BSum(FunctionType type = FunctionType.Whole) : IBasicFunction<SingleMatrix>
    {
        private readonly FunctionType type = type;

        protected override Matrix ApplyCPU(SingleMatrix p)
        {
            MatrixCPU m = (MatrixCPU)p.m.matrixBase;
            float[,] values = new float[
                type == FunctionType.Column ? p.m.shape.width : 1,
                type == FunctionType.Row ? p.m.shape.height : 1];

            switch (type)
            {
                case FunctionType.Column:
                    CPUManager.RunTask(0, p.m.shape.width, row =>
                    {
                        for (int col = 0; col < p.m.shape.height; col++)
                        {
                            float val = m[row, col, p.m.transposed];
                            if (!float.IsNegativeInfinity(val))
                                values[row, 0] += val;
                        }
                    });
                    break;
                case FunctionType.Row:
                    CPUManager.RunTask(0, p.m.shape.height, col =>
                    {
                        for (int row = 0; row < p.m.shape.width; row++)
                        {
                            float val = m[row, col, p.m.transposed];
                            if (!float.IsNegativeInfinity(val))
                                values[0, col] += val;
                        }
                    });
                    break;
                default:
                    for (int row = 0; row < p.m.shape.width; row++)
                        for (int col = 0; col < p.m.shape.height; col++)
                        {
                            float val = m[row, col, p.m.transposed];
                            if (!float.IsNegativeInfinity(val))
                                values[0, 0] += val;
                        }
                    break;
            }
            p.result = new(values, new MatrixOptions() { Device = Device.CPU }, new BackwardContext<SingleMatrix>(this, p));
            return p.result;
        }

        protected override Matrix ApplyGPU(SingleMatrix p)
        {
            throw new NotImplementedException();
        }

        protected override void ApplyGradientCPU(SingleMatrix p)
        {
            MatrixCPU m = (MatrixCPU)p.m.matrixBase;
            MatrixCPU r = (MatrixCPU)p.result!.matrixBase;
            switch (type)
            {
                case FunctionType.Column:
                    CPUManager.RunTask(0, p.m.shape.width, row =>
                    {
                        for (int col = 0; col < p.m.shape.height; col++)
                            if (!float.IsNegativeInfinity(m[row, col, p.m.transposed]))
                                m.AddGradient(row, col, r.GetGradient(row, 0, p.result.transposed), p.m.transposed);
                    });
                    break;
                case FunctionType.Row:
                    CPUManager.RunTask(0, p.m.shape.height, col =>
                    {
                        for (int row = 0; row < p.m.shape.width; row++)
                            if (!float.IsNegativeInfinity(m[row, col, p.m.transposed]))
                                m.AddGradient(row, col, r.GetGradient(0, col, p.result.transposed), p.m.transposed);
                    });
                    break;
                default:
                    float grad = r.GetGradient(0, 0, p.result.transposed);
                    for (int row = 0; row < p.m.shape.width; row++)
                        for (int col = 0; col < p.m.shape.height; col++)
                            if (!float.IsNegativeInfinity(m[row, col, p.m.transposed]))
                                m.AddGradient(row, col, grad, p.m.transposed);
                    break;
            }
        }

        protected override void ApplyGradientGPU(SingleMatrix p)
        {
            throw new NotImplementedException();
        }

        protected override void ValidateParameters(SingleMatrix p) { }
    }
}