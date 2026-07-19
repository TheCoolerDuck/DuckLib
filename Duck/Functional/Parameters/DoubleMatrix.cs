using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functional.Parameters
{
    public class DoubleMatrix(Matrix a, Matrix b) : IParameter
    {
        public readonly Matrix a = a;
        public readonly Matrix b = b;

        public Matrix? result { get; set; }
        public object? cache { get; set; }

        public static implicit operator DoubleMatrix((Matrix a, Matrix b) tuple)
        {
            return new DoubleMatrix(tuple.a, tuple.b);
        }
        public Matrix[] MatricesUsed()
        {
            return [a, b];
        }
    }
}
