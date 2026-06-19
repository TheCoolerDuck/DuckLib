using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Parameters
{
    public class SingleMatrix(Matrix m) : IParameter
    {
        public readonly Matrix m = m;
        public Matrix? result { get; set; }

        public static implicit operator SingleMatrix(Matrix m)
        {
            return new SingleMatrix(m);
        }

        public Matrix[] MatricesUsed()
        {
            return [m];
        }
    }
}
