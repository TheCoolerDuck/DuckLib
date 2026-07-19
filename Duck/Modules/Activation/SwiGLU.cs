using Duck.Modules.Basic;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Modules.Activation
{
    public class SwiGLU : Module
    {
        private readonly Linear a;
        private readonly Linear b;
        private readonly Sigmoid sigmoid;

        public SwiGLU(int input, int output, Module? parent, string name = "SwiGLU") : base(parent, name)
        {
            a = new(input, output, this, name + " linear a");
            b = new(input, output, this, name + " linear b");
            sigmoid = new(parent);
        }
        public override Matrix Forward(Matrix m)
        {
            return a.Forward(m) * sigmoid.Forward(b.Forward(m));
        }

        public override Matrix[] GetParameters()
        {
            return [.. a.GetParameters(), .. b.GetParameters()];
        }
    }
}
