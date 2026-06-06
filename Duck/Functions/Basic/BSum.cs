using Duck.CustomLLM.Library.Objects.MatrixObjects;
using Duck.Functions.Parameters;
using Duck.Management;
using Duck.Matrix_Utilities;

namespace Duck.Functions.Basic
{
    internal class BSum(FunctionType type = FunctionType.Whole) : IBasicFunction<SingleMatrix>
    {
        private readonly FunctionType type = type;

        public Matrix Apply(SingleMatrix p)
        {
            if (p.m.device != Device_Management.Device.CPU)
                throw new NotImplementedException();
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
            p.result = new(values, new BackwardContext<SingleMatrix>(this, p));
            return p.result;
        }

        public void ApplyGradient(SingleMatrix p)
        {
            if (p.result == null)
                throw new ArgumentException("Params must have a result");
            if (p.m.device != Device_Management.Device.CPU)
                throw new NotImplementedException();
            MatrixCPU m = (MatrixCPU)p.m.matrixBase;
            MatrixCPU r = (MatrixCPU)p.result.matrixBase;
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
    }
}