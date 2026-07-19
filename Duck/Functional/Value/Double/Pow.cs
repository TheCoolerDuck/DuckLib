using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Value.Double
{
    public class Pow : IDoubleValueFunction
    {
        public static float Apply(float a, float b)
        {
            return MathF.Pow(a, b);
        }
        public static (float a, float b) ApplyDerivative(float a, float b)
        {
            float p = MathF.Pow(a, b);
            return (b * p / a, p * MathF.Log(a));
        }
        public static string GetGPUApply()
        {
            return "return powf(x, y);";
        }

        public static string GetGPUApplyDerivative()
        {
            return @"
            {
                float p = powf(x, y);
                return make_float2(y * p / x, p * logf(x));
            };";
        }
    }
}
