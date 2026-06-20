using Duck.Modules;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Modules.Basic
{
    public class Linear(int width, int height, string name = "linear") : IModule
    {
        private readonly Matrix weights = new(Matrix.Random(width, height, MathF.Sqrt(width)), new MatrixOptions() { Name = name + "-weight" });
        private readonly Matrix bias = new(Matrix.Random(1, height), new MatrixOptions() { Name = name + "-bias" });

        public Matrix Forward(Matrix m)
        {
            return (m >> weights) + bias;
        }

        public Matrix[] GetParameters()
        {
            return [weights, bias];
        }
    }
}
