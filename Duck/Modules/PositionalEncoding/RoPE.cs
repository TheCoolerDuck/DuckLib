using Duck.Functional.Elementary;
using Duck.Functional.Value.Double;
using Duck.Functions.Value.Single;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Modules.PositionalEncoding
{
    public class RoPE : IModule
    {
        private readonly Matrix[] matrices;
        private readonly Concatenate concatenate = new(FunctionType.Column);
        public RoPE(int size, int maxSequenceLength)
        {
            matrices = new Matrix[maxSequenceLength];

            for (int m = 0; m < maxSequenceLength; m++)
            {
                float[,] values = new float[size, size];

                for (int i = 0; i < size / 2; i++)
                {
                    int a = 2 * i;
                    int b = a + 1;

                    float theta = m * MathF.Pow(1000f, -(2f * i) / size);

                    values[a, a] = MathF.Cos(theta);
                    values[a, b] = MathF.Sin(theta);

                    values[b, a] = -MathF.Sin(theta);
                    values[b, b] = MathF.Cos(theta);
                }

                matrices[m] = new Matrix(values, new MatrixOptions() { HasGrad = false, Name = "RoPE " + m });
            }
        }
        public Matrix Forward(Matrix m)
        {
            Matrix[] output = new Matrix[m.shape.width];

            for (int p = 0; p < m.shape.width; p++)
            {
                output[p] = m.GetRow(p).T() >> matrices[p];
            }

            return concatenate.Apply(output);
        }

        public Matrix[] GetParameters()
        {
            return [];
        }
    }
}
