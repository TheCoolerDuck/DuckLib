using Duck.Functions.Basic;
using Duck.Functions.Value.Single;
using Duck.Modules;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Modules.Activation
{
    public class Sigmoid : IModule
    {
        private static readonly ElementFunction<Exp> exp = new();
        public Matrix Forward(Matrix m)
        {
            return 1 / (1 + exp.Apply(-m));
        }

        public Matrix[] GetParameters()
        {
            return [];
        }
    }
}
