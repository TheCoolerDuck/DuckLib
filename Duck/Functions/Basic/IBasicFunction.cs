using Duck.CustomLLM.Library.Objects.MatrixObjects;
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

    public interface IBasicFunction<T> where T : IParameter
    {
        public abstract Matrix Apply(T p);
        public abstract void ApplyGradient(T p);
    }
}
