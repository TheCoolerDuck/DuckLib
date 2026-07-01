using Duck.Functional.Elementary;
using Duck.Functional.Value.Double;
using Duck.Functions.Value.Single;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Modules.PositionalEncoding
{
    public class BLOS(int size) : IModule
    {
        private readonly Matrix weight = new(Matrix.Random(size, size) * 0.001f);
        private readonly Matrix bias = new(Matrix.Identity(size) * (MathF.PI / 2) + Matrix.Random(size, size) * 0.001f);
        private readonly Concatenate concatenate = new(FunctionType.Column);

        public Matrix Forward(Matrix m)
        {
            Matrix[] output = new Matrix[m.shape.width];
            Matrix r = 2 * MathF.PI / (new ElementFunction<Abs>().Apply(weight) + 0.00001f);

            for (int y = 0; y < m.shape.width; y++)
            {
                Matrix s = new Matrix(new float[,] { { y } });
                Matrix a = new MatrixFunction<Mod>().Apply(Matrix.Broadcast(s, r));

                a.Detach();

                Matrix t = weight * a + bias;
                output[y] = m.GetRow(y).T() >> t;
            }

            return concatenate.Apply(output);
        }

        public Matrix[] GetParameters()
        {
            return [weight, bias];
        }
    }
}
