using Duck.Modules.Basic;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Modules.Activation
{
    public class SwiGLU(int input, int output, string name = "SwiGLU") : IModule
    {
        private readonly Linear a = new(input, output, name + " linear a");
        private readonly Linear b = new(input, output, name + " linear b");
        private readonly Sigmoid sigmoid = new();
        public Matrix Forward(Matrix m)
        {
            return a.Forward(m) * sigmoid.Forward(b.Forward(m));
        }

        public Matrix[] GetParameters()
        {
            return [.. a.GetParameters(), .. b.GetParameters()];
        }
    }
}
