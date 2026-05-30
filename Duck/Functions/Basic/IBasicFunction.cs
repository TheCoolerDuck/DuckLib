using Duck.CustomLLM.Library.Objects.MatrixObjects;
using Duck.Functions.Parameters;
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
        public Matrix Apply(T p);
        public void ApplyGradient(T p);
    }
}
