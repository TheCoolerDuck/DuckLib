using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Value.Double
{
    public class Max : IDoubleValueFunction
    {
        public static float Apply(float a, float b)
        {
            return Math.Max(a, b);
        }
        public static (float a, float b) ApplyDerivative(float a, float b)
        {
            return a >= b ? (1.0f, 0.0f) : (0.0f, 1.0f);
        }
        public static string GetGPUApply()
        {
            return "return x > y ? x : y;";
        }

        public static string GetGPUApplyDerivative()
        {
            return "return x > y ? make_float2(1.0f, 0.0f) : make_float2(0.0f, 1.0f);";
        }
    }
}
