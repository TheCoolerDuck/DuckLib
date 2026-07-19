using Duck.Functions.Value.Double;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functional.Value.Double
{
    public class Mod : IDoubleValueFunction
    {
        public static float Apply(float a, float b)
        {
            return a % b;
        }

        public static (float a, float b) ApplyDerivative(float a, float b)
        {
            return (1, -MathF.Floor(a / b));
        }

        public static string GetGPUApply()
        {
            return "return x - y * floorf(x / y);";
        }

        public static string GetGPUApplyDerivative()
        {
            return "return make_float2(1, -floorf(x / y));";
        }
    }
}
