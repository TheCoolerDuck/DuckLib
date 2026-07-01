
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Value.Single
{
    public interface ISingleValueFunction
    {
        public static abstract float Apply(float v);
        public static abstract float ApplyDerivative(float v);
        internal static abstract string GetGPUApply();
        internal static abstract string GetGPUApplyDerivative();
    }
}
