using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Value.Double
{
    public class Pow : IDoubleValueFunction
    {
        public static double Apply(double a, double b)
        {
            return Math.Pow(a, b);
        }
        public static (double a, double b) ApplyDerivative(double a, double b)
        {
            return (b * Math.Pow(a, b - 1), Math.Pow(a, b) * Math.Log(a));
        }
    }
}
