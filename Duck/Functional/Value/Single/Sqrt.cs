using Duck.Functions.Value.Single;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functional.Value.Single
{
    public class Sqrt : ISingleValueFunction
    {
        public static float Apply(float v)
        {
            return MathF.Sqrt(v);
        }
        public static float ApplyDerivative(float v)
        {
            return 1 / (2 * MathF.Sqrt(v));
        }

        public static string GetGPUApply()
        {
            return "return sqrtf(x);";
        }

        public static string GetGPUApplyDerivative()
        {
            return "return 1.0f / (2.0f * sqrtf(x));";
        }
    }
}
