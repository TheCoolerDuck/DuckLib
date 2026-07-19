using Duck.Functional.Parameters;
using Duck.Management;
using Duck.Matrix_Utilities;
using ManagedCuda;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functional.Loss
{
    public abstract class Loss<T> where T : IParameter
    {
        public Matrix Apply(T p)
        {
            ValidateParameters(p);

            if (p.GetOperationDevice() == Device.CPU)
            {
                (int width, int height) = GetResultShape(p);

                float[,] result = new float[width, height];
                ApplyCPU(p, result);

                return new Matrix(result,
                    new MatrixOptions()
                    {
                        Device = Device.CPU,
                        HasGrad = p.ResultHasGradient()
                    },
                    new BackwardContext<T>(null, p));
            }
            else if (p.GetOperationDevice() == Device.GPU)
            {
                (int width, int height) = GetResultShape(p);
                CudaDeviceVariable<float> values = new(width * height);

                Matrix result = new((width, height), values, new BackwardContext<T>(null, p), p.ResultHasGradient());
                ApplyGPU(p, result);

                return result;
            }
            else
                throw new ArgumentException("Unspecified device for operation");
        }
        protected abstract void ApplyCPU(T p, float[,] result);
        protected abstract void ApplyGPU(T p, Matrix result);
        protected abstract (int width, int height) GetResultShape(T p);
        protected abstract void ValidateParameters(T p);
    }
}
