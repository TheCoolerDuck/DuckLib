using Duck.Functional.Elementary;
using Duck.Functional.Elementary.Concatenate;
using Duck.Functional.Value.Double;
using Duck.Functions.Value.Single;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Modules.PositionalEncoding
{
    public class RoPE : Module
    {
        private readonly Matrix[] matrices;
        public RoPE(int size, int maxSequenceLength, Module? parent = null, string name = "RoPE") : base(parent, name)
        {
            matrices = new Matrix[maxSequenceLength];

            for (int m = 0; m < maxSequenceLength; m++)
            {
                float[,] values = new float[size, size];

                for (int i = 0; i < size / 2; i++)
                {
                    int a = 2 * i;
                    int b = a + 1;

                    float theta = m * MathF.Pow(100000f, -(2f * i) / size);

                    values[a, a] = MathF.Cos(theta);
                    values[a, b] = MathF.Sin(theta);

                    values[b, a] = -MathF.Sin(theta);
                    values[b, b] = MathF.Cos(theta);
                }

                matrices[m] = new Matrix(values, new MatrixOptions() { HasGrad = false, Name = "RoPE " + m });
            }
        }
        public override Matrix Forward(Matrix m)
        {
            Matrix[] output = new Matrix[m.shape.width];

            for (int p = 0; p < m.shape.width; p++)
            {
                output[p] = m.GetRow(p).T() >> matrices[p];
            }

            return new Concatenate(FunctionType.Column, m.shape.width).Apply(output);
        }

        public override Matrix[] GetParameters()
        {
            return [];
        }
    }
}
