using Duck.Functional.Elementary;
using Duck.Functions.Value.Single;
using Duck.Modules;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Modules.Basic
{
    public class Apply<T> : IModule where T : ISingleValueFunction
    {
        private readonly ElementFunction<T> function = new();
        public Matrix Forward(Matrix m)
        {
            return function.Apply(m);
        }

        public Matrix[] GetParameters()
        {
            return [];
        }
    }
}
