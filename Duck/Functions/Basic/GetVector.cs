using Duck.CustomLLM.Library.Objects.MatrixObjects;
using Duck.Functions.Parameters;
using Duck.Management;
using Duck.Matrix_Utilities;

namespace Duck.Functions.Basic
{
    internal class GetVector(FunctionType type) : IBasicFunction<MatrixAndIndex>
    {
        public FunctionType type = Validate(type);

        private static FunctionType Validate(FunctionType t)
        {
            if (t == FunctionType.Whole)
                throw new ArgumentException("FunctionType.Whole is not allowed.", nameof(t));
            return t;
        }

        public Matrix Apply(MatrixAndIndex p)
        {
            if (p.m.device != Device_Management.Device.CPU)
                throw new NotImplementedException();

            bool isRow = type == FunctionType.Row;
            int vectorLen = isRow ? p.m.shape.width : p.m.shape.height;

            double[,] vector = new double[isRow ? vectorLen : 1, isRow ? 1 : vectorLen];
            MatrixCPU m = (MatrixCPU)p.m.matrixBase;

            CPUThreadManager.RunTask(0, vectorLen, i =>
            {
                int srcRow = isRow ? i : p.i;
                int srcCol = isRow ? p.i : i;
                int dstRow = isRow ? i : 0;
                int dstCol = isRow ? 0 : i;
                vector[dstRow, dstCol] = m[srcRow, srcCol, p.m.transposed];
            });

            p.result = new(vector, new BackwardContext<MatrixAndIndex>(this, p));
            return p.result;
        }

        public void ApplyGradient(MatrixAndIndex p)
        {
            if (p.result == null)
                throw new ArgumentException("Params must have a result.");

            bool isRow = type == FunctionType.Row;
            int vectorLen = isRow ? p.m.shape.width : p.m.shape.height;

            MatrixCPU m = (MatrixCPU)p.m.matrixBase;
            MatrixCPU r = (MatrixCPU)p.result.matrixBase;

            CPUThreadManager.RunTask(0, vectorLen, i =>
            {
                int srcRow = isRow ? i : p.i;
                int srcCol = isRow ? p.i : i;
                int dstRow = isRow ? i : 0;
                int dstCol = isRow ? 0 : i;
                m.AddGradient(srcRow, srcCol, r.GetGradient(dstRow, dstCol, p.result.transposed), p.m.transposed);
            });
        }
    }
}