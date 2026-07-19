using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Value.Single
{
    public class Abs : ISingleValueFunction
    {
        public static float Apply(float v)
        {
            return MathF.Abs(v);
        }
        public static float ApplyDerivative(float v)
        {
            return v < 0 ? -1 : 1;
        }

        public static string GetGPUApply()
        {
            return "return x < 0 ? -x : x;";
        }

        public static string GetGPUApplyDerivative()
        {
            return "return x < 0 ? -1 : 1;";
        }
    }
}
