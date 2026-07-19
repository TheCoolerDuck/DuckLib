using Duck.Functional.Parameters;
using Duck.Management;
using Duck.Matrix_Utilities;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using System.Threading.Tasks;

namespace Duck.Functional.Elementary.Concatenate
{
    public class Concatenate : BasicFunction<MatrixArray>
    {
        public readonly FunctionType type;
        private readonly CudaDeviceVariable<GPUMatrixStruct>? matrixArrayGPU;

        public Concatenate(FunctionType type, int itemCount)
        {
            if (type == FunctionType.Whole)
                throw new ArgumentException("FunctionType.Whole is not allowed.", nameof(type));

            this.type = type;

            if (GPUManager.live)
            {
                CudaDeviceVariable<GPUMatrixStruct> gpuValues = new(itemCount);
                matrixArrayGPU = gpuValues;
            }
        }

        protected override void ApplyCPU(MatrixArray p, float[,] result)
        {
            (int width, int height) = GetResultShape(p);

            MatrixCPU[] srcs = [.. p.a.Select(m => (MatrixCPU)m.matrixBase)];

            CPUManager.RunTask(0, width, 0, height, (x, y) =>
            {
                int idx = 0;
                int s = type == FunctionType.Row ? y : x;

                while (s >= (type == FunctionType.Row ? p.a[idx].shape.height : p.a[idx].shape.width))
                    s -= type == FunctionType.Row ? p.a[idx++].shape.height : p.a[idx++].shape.width;

                MatrixCPU src = srcs[idx];
                int sx = type == FunctionType.Column ? s : x;
                int sy = type == FunctionType.Row ? s : y;
                result[x, y] = src[sx, sy, p.a[idx].transposed];
            });
        }

        protected override void ApplyGPU(MatrixArray p, Matrix result)
        {
            GPUManager.SetKernelSize(gradientKernel, result.size);

            if (type == FunctionType.Column)
                for (int i = 0; i < p.a.Length; i++)
                    p.a[i] = p.a[i].T();

            matrixArrayGPU!.CopyToDevice([.. p.a.Select(m => m.GPUValues())]);

            applyKernel.Run(matrixArrayGPU.DevicePointer, (type == FunctionType.Column ? result.T() : result).GPUValues());
        }

        protected override void ApplyGradientCPU(MatrixArray p)
        {
            if (p.result == null)
                throw new ArgumentException("Params must have a result");
            if (p.a[0].device != Device.CPU)
                throw new NotImplementedException();

            MatrixCPU[] dsts = [.. p.a.Select(m => (MatrixCPU)m.matrixBase)];
            MatrixCPU r = (MatrixCPU)p.result.matrixBase;

            CPUManager.RunTask(0, p.result.shape.width, 0, p.result.shape.height, (x, y) =>
            {
                int idx = 0;
                int s = type == FunctionType.Row ? y : x;

                while (s >= (type == FunctionType.Row ? p.a[idx].shape.height : p.a[idx].shape.width))
                    s -= type == FunctionType.Row ? p.a[idx++].shape.height : p.a[idx++].shape.width; 
                
                MatrixCPU dst = dsts[idx];
                int sx = type == FunctionType.Column ? s : x;
                int sy = type == FunctionType.Row ? s : y;
                dst.AddGradient(sx, sy, r.GetGradient(x, y, p.result.transposed), p.a[idx].transposed);
            });
        }

        protected override void ApplyGradientGPU(MatrixArray p)
        {
            GPUManager.SetKernelSize(gradientKernel, p.result!.size);

            if (type == FunctionType.Column)
                for (int i = 0; i < p.a.Length; i++)
                    p.a[i] = p.a[i].T();

            matrixArrayGPU!.CopyToDevice([.. p.a.Select(m => m.GPUGradient())]);

            gradientKernel.Run(matrixArrayGPU.DevicePointer, (type == FunctionType.Column ? p.result.T() : p.result).GPUGradient());
        }

        protected override (int width, int height) GetResultShape(MatrixArray p)
        {
            int nw = type == FunctionType.Column ? p.a.Sum(m => m.shape.width) : p.a[0].shape.width;
            int nh = type == FunctionType.Row ? p.a.Sum(m => m.shape.height) : p.a[0].shape.height;

            return (nw, nh);
        }

        protected override void ValidateParameters(MatrixArray p)
        {
            if (p.a == null || p.a.Length == 0)
                throw new ArgumentException("Matrix array must not be empty.");

            for (int i = 1; i < p.a.Length; i++)
            {
                if (type == FunctionType.Row && p.a[i].shape.width != p.a[0].shape.width)
                    throw new ArgumentException($"Matrix shapes incompatible for row concatenation at index {i}: {p.a[i].shape} | {p.a[0].shape}");
                if (type == FunctionType.Column && p.a[i].shape.height != p.a[0].shape.height)
                    throw new ArgumentException($"Matrix shapes incompatible for column concatenation at index {i}: {p.a[i].shape} | {p.a[0].shape}");
            }
        }

        private static readonly CudaKernel applyKernel = GPUManager.Compile();
        private static readonly CudaKernel gradientKernel = GPUManager.CompileGradient();
    }
}