using Duck.Modules;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Modules.Activation
{
    public class Swish : IModule
    {
        private static readonly Sigmoid sigmoid = new();
        public Matrix Forward(Matrix m)
        {
            return sigmoid.Forward(m) * m;
        }

        public Matrix[] GetParameters()
        {
            return [];
        }
    }
}
