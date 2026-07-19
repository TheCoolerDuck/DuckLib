using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Value.Single
{
    public class Cos : ISingleValueFunction
    {
        public static float Apply(float v)
        {
            return MathF.Cos(v);
        }
        public static float ApplyDerivative(float v)
        {
            return -MathF.Sin(v);
        }
        public static string GetGPUApply()
        {
            return "return cosf(x);";
        }

        public static string GetGPUApplyDerivative()
        {
            return "return -sinf(x);";
        }
    }
}
