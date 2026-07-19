using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Value.Single
{
    public class Exp : ISingleValueFunction
    {
        public static float Apply(float v)
        {
            return MathF.Exp(v);
        }

        public static float ApplyDerivative(float v)
        {
            return MathF.Exp(v);
        }

        public static string GetGPUApply()
        {
            return "return expf(x);";
        }

        public static string GetGPUApplyDerivative()
        {
            return "return expf(x);";
        }
    }
}
