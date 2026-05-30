using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Value.Single
{
    public class Cos : ISingleValueFunction
    {
        public static double Apply(double v)
        {
            return Math.Cos(v);
        }
        public static double ApplyDerivative(double v)
        {
            return -Math.Sin(v);
        }
    }
}
