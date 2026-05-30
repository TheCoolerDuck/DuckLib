using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Value.Single
{
    public class Tan : ISingleValueFunction
    {
        public static double Apply(double v)
        {
            return Math.Tan(v);
        }
        public static double ApplyDerivative(double v)
        {
            double c = Math.Cos(v);
            return 1.0 / (c * c);
        }
    }
}
