using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Value.Single
{
    public class Tan : ISingleValueFunction
    {
        public static float Apply(float v)
        {
            return MathF.Tan(v);
        }
        public static float ApplyDerivative(float v)
        {
            float c = MathF.Cos(v);
            return 1.0f / (c * c);
        }

        public static string GetGPUApply()
        {
            return "tanf(x)";
        }

        public static string GetGPUApplyDerivative()
        {
            return "1.0f / (cosf(x) * cosf(x))";
        }
    }
}
