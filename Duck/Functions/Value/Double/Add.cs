using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Value.Double
{
    public class Add : IDoubleValueFunction
    {
        public static double Apply(double a, double b)
        {
            return a + b;
        }

        public static (double a, double b) ApplyDerivative(double a, double b)
        {
            return (1, 1);
        }
    }
}
