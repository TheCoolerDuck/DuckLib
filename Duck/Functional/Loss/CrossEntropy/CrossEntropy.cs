using Duck.Functional.Elementary;
using Duck.Functional.Parameters;
using Duck.Management;
using Duck.Matrix_Utilities;
using Duck.Modules.Basic;
using ManagedCuda;

namespace Duck.Functional.Loss.CrossEntropy
{
    public class CrossEntropy(int sequenceSize) : Loss<MatrixAndIndexArray>
    {
        private readonly static SoftMax softMax = new(null);
        private readonly CudaDeviceVariable<int> truths = new(sequenceSize);
        protected override void ApplyCPU(MatrixAndIndexArray p, float[,] result)
        {
            Matrix probs = softMax.Forward(p.m.CloneValues());

            MatrixCPU m = (MatrixCPU)p.m.matrixBase;
            MatrixCPU pro = (MatrixCPU)probs.matrixBase;

            CPUManager.RunTask(0, probs.shape.width, i =>
            {
                result[i, 0] = -MathF.Log(pro[i, p.i[i], probs.transposed] + 1e-8f);
            });

            CPUManager.RunTask(0, p.m.shape.width, 0, p.m.shape.height, (i, j) =>
            {
                bool isHot = j == p.i[i];
                m.AddGradient(i, j, isHot ? pro[i, j, probs.transposed] - 1 : pro[i, j, probs.transposed], p.m.transposed);
            });
        }

        protected override void ApplyGPU(MatrixAndIndexArray p, Matrix result)
        {
            Matrix probs = softMax.Forward(p.m.CloneValues());

            truths.CopyToDevice(p.i);

            GPUManager.SetKernelSize(applyKernel, result.size);

            applyKernel.Run(probs.GPUValues(), truths.DevicePointer, result.GPUValues());

            GPUManager.SetKernelSize(applyKernel, p.m.size);

            gradientKernel.Run(p.m.GPUGradient(), probs.GPUValues(), truths.DevicePointer);
        }

        protected override (int width, int height) GetResultShape(MatrixAndIndexArray p)
        {
            return (p.m.shape.width, 1);
        }

        protected override void ValidateParameters(MatrixAndIndexArray p)
        {
            if (p.m.shape.width != p.i.Length)
                throw new ArgumentException($"Matrix and Array must have same width/length {p.m} {p.i}");

            for (int i = 0; i < p.m.shape.width; i++)
                if (p.i[i] < 0 || p.i[i] >= p.m.shape.height)
                    throw new ArgumentException($"Invalid truth selection at index: {i} of {p.i[i]}");
        }

        private static readonly CudaKernel applyKernel = GPUManager.Compile();
        private static readonly CudaKernel gradientKernel = GPUManager.CompileGradient();
    }
}