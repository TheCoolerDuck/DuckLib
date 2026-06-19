using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Value.Double
{
    public class Mul : IDoubleValueFunction
    {
        public static float Apply(float a, float b)
        {
            return a * b;
        }
        public static (float a, float b) ApplyDerivative(float a, float b)
        {
            return (b, a);
        }
        public static string GetGPUApply()
        {
            return "x * y";
        }

        public static string GetGPUApplyDerivative()
        {
            return "make_float2(y, x)";
        }
    }
}
