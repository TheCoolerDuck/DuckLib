using Duck.CustomLLM.Library.Objects.MatrixObjects;
using Duck.Functions.Basic;
using Duck.Modules;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Modules.Advanced
{
    public class BLOS(int size) : IModule
    {
        private readonly Matrix weight = new(Matrix.Zeros(size, size));
        private readonly Matrix bias = new(Matrix.Random(size, size));
        private readonly Concatenate concatenate = new(FunctionType.Column);

        public Matrix Forward(Matrix m)
        {
            Matrix[] output = new Matrix[m.shape.width];
            for (int y = 0; y < m.shape.width; y++)
            {
                Matrix t = weight * y + bias;
                output[y] = t << m.getCol(y);
            }

            return concatenate.Apply(output);
        }

        public Matrix[] GetParameters()
        {
            return [weight, bias];
        }
    }
}
