using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Value.Single
{
    public class Log : ISingleValueFunction
    {
        public static float Apply(float v)
        {
            return MathF.Log(v);
        }
        public static float ApplyDerivative(float v)
        {
            return 1.0f / v;
        }

        public static string GetGPUApply()
        {
            return "return logf(x);";
        }

        public static string GetGPUApplyDerivative()
        {
            return "return 1.0f / x;";
        }
    }
}
