using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Parameters
{
    public class MatrixAndIndex(Matrix m, int i) : IParameter
    {
        public readonly Matrix m = m;
        public readonly int i = i;
        public Matrix? result { get; set; }

        public static implicit operator MatrixAndIndex((Matrix m, int i) tuple)
        {
            return new MatrixAndIndex(tuple.m, tuple.i);
        }
        public Matrix[] MatricesUsed()
        {
            return [m];
        }
    }
}
