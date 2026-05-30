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
        public static abstract double Apply(double a, double b);
        public static abstract (double a, double b) ApplyDerivative(double a, double b);
    }
}
