using Duck.Modules;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Modules.Activation
{
    public class Swish : Module
    {
        private readonly Sigmoid sigmoid;
        public Swish(Module? parent, string name = "Swish") : base(parent, name)
        {
            sigmoid = new(this);
        }
        public override Matrix Forward(Matrix m)
        {
            return sigmoid.Forward(m) * m;
        }

        public override Matrix[] GetParameters()
        {
            return [];
        }
    }
}
