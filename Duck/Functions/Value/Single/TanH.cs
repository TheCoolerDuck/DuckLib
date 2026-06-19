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

        public static string GetGPUApply()
        {
            return "tanhf(x)";
        }

        public static string GetGPUApplyDerivative()
        {
            return "1.0f - tanhf(x) * tanhf(x)";
        }
    }
}
