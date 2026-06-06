using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Value.Single
{
    public class TanH : ISingleValueFunction
    {
        public static float Apply(float v)
        {
            return MathF.Tanh(v);
        }

        public static float ApplyDerivative(float v)
        {
            return MathF.Pow(1 / MathF.Cosh(v), 2);
        }
    }
}
