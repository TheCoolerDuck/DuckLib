using Duck.Functions.Parameters;
using Duck.Management;
using ManagedCuda;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Basic
{
    public enum FunctionType
    {
        Whole,
        Row,
        Column
    }

    public abstract class IBasicFunction<T> where T : IParameter
    {
        public Matrix Apply(T p)
        {
            ValidateParameters(p);

            return p.GetOperationDevice() switch
            {
                Device.CPU => ApplyCPU(p),
                Device.GPU => ApplyGPU(p),
                _ => throw new ArgumentException("Unspecified device for operation"),
            };
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
        protected abstract Matrix ApplyCPU(T p);
        protected abstract void ApplyGradientCPU(T p);
        protected abstract Matrix ApplyGPU(T p);
        protected abstract void ApplyGradientGPU(T p);
        protected abstract void ValidateParameters(T p);
    }
}
