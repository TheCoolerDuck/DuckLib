using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Value.Double
{
    public class Min : IDoubleValueFunction
    {
        public static double Apply(double a, double b)
        {
            return Math.Min(a, b);
        }
        public static (double a, double b) ApplyDerivative(double a, double b)
        {
            return a <= b ? (1.0, 0.0) : (0.0, 1.0);
        }
    }
}
