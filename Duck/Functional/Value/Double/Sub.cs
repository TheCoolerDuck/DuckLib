using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Value.Double
{
    public class Sub : IDoubleValueFunction
    {
        public static float Apply(float a, float b)
        {
            return a - b;
        }

        public static (float a, float b) ApplyDerivative(float a, float b)
        {
            return (1, -1);
        }
        public static string GetGPUApply()
        {
            return "return x - y;";
        }

        public static string GetGPUApplyDerivative()
        {
            return "return make_float2(1.0f, -1.0f);";
        }
    }
}
