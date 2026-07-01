using Duck.Functional.Parameters;
using Duck.Functions.Parameters;
using Duck.Management;
using Duck.Matrix_Utilities;
using ManagedCuda;

namespace Duck.Functional.Elementary
{
    public class GetVectors(FunctionType type) : IBasicFunction<MatrixAndIndexArray>
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
            bool isCol = type == FunctionType.Column;
            int vectorLen = isCol ? p.m.shape.width : p.m.shape.height;

            int outWidth = isCol ? p.i.Length : vectorLen;
            int outHeight = isCol ? vectorLen : p.i.Length;

            float[,] vector = new float[outWidth, outHeight];
            MatrixCPU m = (MatrixCPU)p.m.matrixBase;

            CPUManager.RunTask(0, vectorLen * p.i.Length, i =>
            {
                int srcRow = isCol ? i % vectorLen : p.i[i / vectorLen];
                int srcCol = isCol ? p.i[i / vectorLen] : i % vectorLen;
                int dstRow = isCol ? i / vectorLen : i % vectorLen;
                int dstCol = isCol ? i % vectorLen : i / vectorLen;
                vector[dstRow, dstCol] = m[srcRow, srcCol, p.m.transposed];
            });

            p.result = new(vector, new MatrixOptions() { Device = Device.CPU, HasGrad = ((IParameter)p).ResultHasGradient() }, new BackwardContext<MatrixAndIndexArray>(this, p));

            return p.result;
        }

        protected override Matrix ApplyGPU(MatrixAndIndexArray p)
        {
            bool isCol = type == FunctionType.Column;
            int vectorLen = isCol ? p.m.shape.width : p.m.shape.height;

            int outWidth = isCol ? p.i.Length : vectorLen;
            int outHeight = isCol ? vectorLen : p.i.Length;

            int size = outWidth * outHeight;
            CudaDeviceVariable<float> values = new(size);
            p.result = new Matrix((outWidth, outHeight), values, new BackwardContext<MatrixAndIndexArray>(this, p), ((IParameter)p).ResultHasGradient());

            GPUMatrixStruct a = p.m.GPUValues();
            GPUMatrixStruct r = p.result.GPUValues();

            CudaDeviceVariable<int> indcies = new(p.i.Length);
            indcies.CopyToDevice(p.i);

            applyKernel.BlockDimensions = 64;
            applyKernel.GridDimensions = (size + 63) / 64;

            applyKernel.Run(isCol, a, r, indcies.DevicePointer);

            return p.result;
        }

        protected override void ApplyGradientCPU(MatrixAndIndexArray p)
        {
            bool isCol = type == FunctionType.Column;
            int vectorLen = isCol ? p.m.shape.width : p.m.shape.height;

            MatrixCPU m = (MatrixCPU)p.m.matrixBase;
            MatrixCPU r = (MatrixCPU)p.result!.matrixBase;

            CPUManager.RunTask(0, vectorLen * p.i.Length, i =>
            {
                int srcRow = isCol ? i % vectorLen : p.i[i / vectorLen];
                int srcCol = isCol ? p.i[i / vectorLen] : i % vectorLen;
                int dstRow = isCol ? i / vectorLen : i % vectorLen;
                int dstCol = isCol ? i % vectorLen : i / vectorLen;
                m.AddGradient(srcRow, srcCol, r.GetGradient(dstRow, dstCol, p.result.transposed), p.m.transposed);
            });
        }

        protected override void ApplyGradientGPU(MatrixAndIndexArray p)
        {
            throw new NotImplementedException();
        }

        protected override void ValidateParameters(MatrixAndIndexArray p) { }

        private static CudaKernel? _applyKernel;
        public static CudaKernel applyKernel => _applyKernel ??= GPUManager.Compile(File.ReadAllText("Functional\\GPUCode\\GetVectors.cu"));

        private static CudaKernel? _gradientKernel;
        public static CudaKernel gradientKernel => _gradientKernel ??= GPUManager.Compile(File.ReadAllText("Functional\\GPUCode\\"));
    }
}