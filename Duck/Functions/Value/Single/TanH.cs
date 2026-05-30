using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Value.Single
{
    public class TanH : ISingleValueFunction
    {
        public static double Apply(double v)
        {
            return Math.Tanh(v);
        }

        public static double ApplyDerivative(double v)
        {
            return Math.Pow(1 / Math.Cosh(v), 2);
        }
    }
}
