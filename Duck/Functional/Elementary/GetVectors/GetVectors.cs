using Duck.Functional.Parameters;
using Duck.Management;
using Duck.Matrix_Utilities;
using ManagedCuda;

namespace Duck.Functional.Elementary.GetVectors
{
    public class GetVectors(FunctionType type) : BasicFunction<MatrixAndIndexArray>
    {
        public FunctionType type = Validate(type);

        private static FunctionType Validate(FunctionType t)
        {
            if (t == FunctionType.Whole)
                throw new ArgumentException("FunctionType.Whole is not allowed.", nameof(t));
            return t;
        }
        protected override void ApplyCPU(MatrixAndIndexArray p, float[,] result)
        {
            bool isCol = type == FunctionType.Column;
            int vectorLen = isCol ? p.m.shape.width : p.m.shape.height;

            MatrixCPU m = (MatrixCPU)p.m.matrixBase;

            CPUManager.RunTask(0, vectorLen, 0, p.i.Length, (offset, vector) =>
            {
                int srcRow = isCol ? offset : p.i[vector];
                int srcCol = isCol ? p.i[vector] : offset;

                int dstRow = isCol ? vector : offset;
                int dstCol = isCol ? offset : vector;

                result[dstRow, dstCol] = m[srcRow, srcCol, p.m.transposed];
            });
        }

        protected override void ApplyGPU(MatrixAndIndexArray p, Matrix result)
        {
            bool isCol = type == FunctionType.Column;
            int vectorLen = isCol ? p.m.shape.width : p.m.shape.height;

            int outWidth = isCol ? p.i.Length : vectorLen;
            int outHeight = isCol ? vectorLen : p.i.Length;

            int size = outWidth * outHeight;

            GPUMatrixStruct a = p.m.GPUValues();
            GPUMatrixStruct r = result.GPUValues();

            CudaDeviceVariable<int> indcies = new(p.i.Length);
            indcies.CopyToDevice(p.i);

            GPUManager.SetKernelSize(applyKernel, size);

            applyKernel.Run(isCol, a, r, indcies.DevicePointer);
        }

        protected override void ApplyGradientCPU(MatrixAndIndexArray p)
        {
            bool isCol = type == FunctionType.Column;
            int vectorLen = isCol ? p.m.shape.width : p.m.shape.height;

            MatrixCPU m = (MatrixCPU)p.m.matrixBase;
            MatrixCPU r = (MatrixCPU)p.result!.matrixBase;

            CPUManager.RunTask(0, vectorLen, 0, p.i.Length, (offset, vector) =>
            {
                int srcRow = isCol ? offset : p.i[vector];
                int srcCol = isCol ? p.i[vector] : offset;

                int dstRow = isCol ? vector : offset;
                int dstCol = isCol ? offset : vector;

                m.AddGradient(
                    srcRow,
                    srcCol,
                    r.GetGradient(dstRow, dstCol, p.result.transposed),
                    p.m.transposed);
            });
        }

        protected override void ApplyGradientGPU(MatrixAndIndexArray p)
        {
            throw new NotImplementedException();
        }

        protected override void ValidateParameters(MatrixAndIndexArray p) { }

        protected override (int width, int height) GetResultShape(MatrixAndIndexArray p)
        {
            bool isCol = type == FunctionType.Column;
            int vectorLen = isCol ? p.m.shape.width : p.m.shape.height;

            int outWidth = isCol ? p.i.Length : vectorLen;
            int outHeight = isCol ? vectorLen : p.i.Length;

            return (outWidth, outHeight);
        }

        private readonly static CudaKernel applyKernel = GPUManager.Compile();
        //private readonly static CudaKernel gradientKernel = GPUManager.CompileGradient();
    }
}