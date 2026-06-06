using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Value.Double
{
    public class Add : IDoubleValueFunction
    {
        public static float Apply(float a, float b)
        {
            return a + b;
        }

        public static (float a, float b) ApplyDerivative(float a, float b)
        {
            return (1, 1);
        }
    }
}
