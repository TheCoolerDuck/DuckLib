using Duck.Functional.Elementary.ElementFunction;
using Duck.Functions.Value.Single;
using Duck.Modules;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Modules.Activation
{
    public class Sigmoid(Module? parent, string name = "Sigmoid") : Module(parent, name)
    {
        private static readonly ElementFunction<Exp> exp = new();
        public override Matrix Forward(Matrix m)
        {
            return 1 / (1 + exp.Apply(-m));
        }

        public override Matrix[] GetParameters()
        {
            return [];
        }
    }
}
