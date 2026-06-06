using Duck.CustomLLM.Library.Objects.MatrixObjects;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Value.Double
{
    public interface IDoubleValueFunction
    {
        public static abstract float Apply(float a, float b);
        public static abstract (float a, float b) ApplyDerivative(float a, float b);
    }
}
