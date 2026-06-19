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

        protected override Matrix ApplyCPU(MatrixAndIndexArray p)
        {
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

            p.result = new(vector, new MatrixOptions() { Device = Device.CPU }, new BackwardContext<MatrixAndIndexArray>(this, p));

            return p.result;
        }

        protected override Matrix ApplyGPU(MatrixAndIndexArray p)
        {
            throw new NotImplementedException();
        }

        protected override void ApplyGradientCPU(MatrixAndIndexArray p)
        {
            bool isRow = type == FunctionType.Row;
            int vectorLen = isRow ? p.m.shape.width : p.m.shape.height;

            MatrixCPU m = (MatrixCPU)p.m.matrixBase;
            MatrixCPU r = (MatrixCPU)p.result!.matrixBase;

            CPUManager.RunTask(0, vectorLen * p.i.Length, i =>
            {
                int srcRow = isRow ? i % vectorLen : p.i[i / vectorLen];
                int srcCol = isRow ? p.i[i / vectorLen] : i % vectorLen;
                int dstRow = isRow ? i % vectorLen : 0;
                int dstCol = isRow ? 0 : i % vectorLen;
                m.AddGradient(srcRow, srcCol, r.GetGradient(dstRow, dstCol, p.result.transposed), p.m.transposed);
            });
        }

        protected override void ApplyGradientGPU(MatrixAndIndexArray p)
        {
            throw new NotImplementedException();
        }
    }
}