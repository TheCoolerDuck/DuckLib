using Duck.CustomLLM.Library.Objects.MatrixObjects;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Value.Single
{
    public interface ISingleValueFunction
    {
        public static abstract double Apply(double v);
        public static abstract double ApplyDerivative(double v);
    }
}
