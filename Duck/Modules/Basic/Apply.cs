using Duck.Functional.Elementary.ElementFunction;
using Duck.Functions.Value.Single;
using Duck.Modules;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Modules.Basic
{
    public class Apply<T>(Module? parent, string name = "Apply") : Module(parent, name) where T : ISingleValueFunction
    {
        private readonly ElementFunction<T> function = new();
        public override Matrix Forward(Matrix m)
        {
            return function.Apply(m);
        }

        public override Matrix[] GetParameters()
        {
            return [];
        }
    }
}
