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
            return (b * MathF.Pow(a, b - 1), MathF.Pow(a, b) * MathF.Log(a));
        }
    }
}
