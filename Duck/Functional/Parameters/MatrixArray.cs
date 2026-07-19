using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functional.Parameters
{
    public class MatrixArray(Matrix[] a) : IParameter
    {
        public readonly Matrix[] a = a;
        public Matrix? result { get; set; }
        public object? cache { get; set; }

        public static implicit operator MatrixArray(Matrix[] a)
        {
            return new MatrixArray(a);
        }

        public Matrix[] MatricesUsed()
        {
            return a;
        }
    }
}
