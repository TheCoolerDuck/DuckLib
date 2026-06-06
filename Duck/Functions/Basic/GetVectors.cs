using Duck.CustomLLM.Library.Objects.MatrixObjects;
using Duck.Functions.Parameters;
using Duck.Management;
using Duck.Matrix_Utilities;

namespace Duck.Functions.Basic
{
    internal class GetVectors(FunctionType type) : IBasicFunction<MatrixAndIndexArray>
    {
        public FunctionType type = Validate(type);

        private static FunctionType Validate(FunctionType t)
        {
            if (t == FunctionType.Whole)
                throw new ArgumentException("FunctionType.Whole is not allowed.", nameof(t));
            return t;
        }

        public Matrix Apply(MatrixAndIndexArray p)
        {
            if (p.m.device != Device_Management.Device.CPU)
                throw new NotImplementedException();

            bool isRow = type == FunctionType.Row;
            int vectorLen = isRow ? p.m.shape.width : p.m.shape.height;

            float[,] vector = new float[isRow ? vectorLen : p.i.Length, isRow ? p.i.Length : vectorLen];
            MatrixCPU m = (MatrixCPU)p.m.matrixBase;

            CPUManager.RunTask(0, vectorLen * p.i.Length, i =>
            {
                int srcRow = isRow ? i % vectorLen : p.i[i / vectorLen];
                int srcCol = isRow ? p.i[i / vectorLen] : i % vectorLen;
                int dstRow = isRow ? i % vectorLen : 0;
                int dstCol = isRow ? 0 : i % vectorLen;
                vector[dstRow, dstCol] = m[srcRow, srcCol, p.m.transposed];
            });

            p.result = new(vector, new BackwardContext<MatrixAndIndexArray>(this, p));
            return p.result;
        }

        public void ApplyGradient(MatrixAndIndexArray p)
        {
            if (p.result == null)
                throw new ArgumentException("Params must have a result.");

            bool isRow = type == FunctionType.Row;
            int vectorLen = isRow ? p.m.shape.width : p.m.shape.height;

            MatrixCPU m = (MatrixCPU)p.m.matrixBase;
            MatrixCPU r = (MatrixCPU)p.result.matrixBase;

            CPUManager.RunTask(0, vectorLen * p.i.Length, i =>
            {
                int srcRow = isRow ? i % vectorLen : p.i[i / vectorLen];
                int srcCol = isRow ? p.i[i / vectorLen] : i % vectorLen;
                int dstRow = isRow ? i % vectorLen : 0;
                int dstCol = isRow ? 0 : i % vectorLen;
                m.AddGradient(srcRow, srcCol, r.GetGradient(dstRow, dstCol, p.result.transposed), p.m.transposed);
            });
        }
    }
}