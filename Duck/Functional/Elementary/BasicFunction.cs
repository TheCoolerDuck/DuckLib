using Duck.Functional.Parameters;
using Duck.Management;
using Duck.Matrix_Utilities;
using ManagedCuda;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functional.Elementary
{
    public enum FunctionType
    {
        Whole,
        Row,
        Column
    }

    public abstract class BasicFunction<T> where T : IParameter
    {
        public Matrix Apply(T p)
        {
            ValidateParameters(p);

            if (p.GetOperationDevice() == Device.CPU)
            {
                (int width, int height) = GetResultShape(p);

                float[,] values = new float[width, height];
                ApplyCPU(p, values);

                Matrix result = new(values,
                    new MatrixOptions()
                    {
                        Device = Device.CPU,
                        HasGrad = p.ResultHasGradient()
                    },
                    new BackwardContext<T>(this, p));

                p.result = result;

                return result;
            }
            else if (p.GetOperationDevice() == Device.GPU)
            {
                (int width, int height) = GetResultShape(p);
                CudaDeviceVariable<float> values = new(width * height);

                Matrix result = new((width, height), values, new BackwardContext<T>(this, p), p.ResultHasGradient());
                ApplyGPU(p, result);

                p.result = result;

                return result;
            }
            else
                throw new ArgumentException("Unspecified device for operation");
        }
        public void ApplyGradient(T p)
        {
            if (p.result == null)
                throw new ArgumentException("Parameters must have a result for gradient passes");


            switch (p.GetOperationDevice())
            {
                case Device.CPU: ApplyGradientCPU(p); break;
                case Device.GPU: ApplyGradientGPU(p); break;
                default: throw new ArgumentException("Unspecified device for operation");
            };
        }
        protected abstract void ApplyCPU(T p, float[,] result);
        protected abstract void ApplyGradientCPU(T p);
        protected abstract void ApplyGPU(T p, Matrix result);
        protected abstract void ApplyGradientGPU(T p);
        protected abstract (int width, int height) GetResultShape(T p);
        protected abstract void ValidateParameters(T p);
    }
}
