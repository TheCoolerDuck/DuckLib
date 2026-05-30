using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Value.Single
{
    public class Log : ISingleValueFunction
    {
        public static double Apply(double v)
        {
            return Math.Log(v);
        }
        public static double ApplyDerivative(double v)
        {
            return 1.0 / v;
        }
    }
}
