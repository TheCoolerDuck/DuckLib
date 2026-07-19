using Duck.Functional.Parameters;
using Duck.Management;
using Duck.Matrix_Utilities;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functional.Elementary.Mask
{
    public class Mask : BasicFunction<SingleMatrix> 
    {
        private readonly float maskValue;
        private readonly bool[,] mask;
        private readonly CUdeviceptr? gpuMask;

        public Mask(float maskValue, bool[,] mask)
        {
            this.maskValue = maskValue;
            this.mask = mask;

            if (GPUManager.live)
            {
                //maybe convert bools from 1 bit ints to 32 bit ints to save spac
                int[] mask1d = new int[mask.Length];

                int index = 0;
                for (int i = 0; i < mask.GetLength(0); i++)
                {
                    for (int j = 0; j < mask.GetLength(1); j++)
                    {
                        mask1d[index++] = mask[i, j] ? 1 : 0;
                    }
                }

                CudaDeviceVariable<int> gpuValues = new(mask.Length);
                gpuValues.CopyToDevice(mask1d);

                gpuMask = gpuValues.DevicePointer;
            }
        }

        protected override void ApplyCPU(SingleMatrix p, float[,] result)
        {
            MatrixCPU m = (MatrixCPU)p.m.matrixBase;

            CPUManager.RunTask(0, p.m.shape.width, 0, p.m.shape.height, (x, y) =>
            {
                if (mask[x, y])
                    result[x, y] = maskValue;
                else
                    result[x, y] = m[x, y, p.m.transposed];
            });
        }

        protected override void ApplyGPU(SingleMatrix p, Matrix result)
        {
            GPUManager.SetKernelSize(applyKernel, result.size);

            applyKernel.Run(p.m.GPUValues(), gpuMask, maskValue, result.GPUValues());
        }

        protected override void ApplyGradientCPU(SingleMatrix p)
        {
            MatrixCPU m = (MatrixCPU)p.m.matrixBase;
            MatrixCPU r = (MatrixCPU)p.result!.matrixBase;

            CPUManager.RunTask(0, p.m.shape.width, 0, p.m.shape.height, (x, y) =>
            {
                if (!mask[x, y])
                    m.AddGradient(x, y, r.GetGradient(x, y, p.result.transposed), p.m.transposed);
            });
        }

        protected override void ApplyGradientGPU(SingleMatrix p)
        {
            throw new NotImplementedException();
        }

        protected override (int width, int height) GetResultShape(SingleMatrix p)
        {
            return p.m.shape;
        }

        protected override void ValidateParameters(SingleMatrix p)
        {
            if (p.m.shape.width != mask.GetLength(0) || p.m.shape.height != mask.GetLength(1))
                throw new ArgumentException($"matrix shape must match mask shape {p.m} {(mask.GetLength(0), mask.GetLength(1))}");
        }

        private readonly static CudaKernel applyKernel = GPUManager.Compile();

        public static bool[,] TriL(int size)
        {
            bool[,] result = new bool[size, size];

            for (int x = 0; x < size; x++)
                for (int y = 0; y < size; y++)
                    result[x, y] = x < y;

            return result;
        }
    }
}
