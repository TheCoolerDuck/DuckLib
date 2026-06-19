using Duck.Functions.Basic;
using Duck.Modules;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Modules.Basic
{
    public class Sum(FunctionType type) : IModule
    {
        private readonly BSum sum = new(type);
        public Matrix Forward(Matrix m)
        {
            return sum.Apply(m);
        }
        public Matrix[] GetParameters()
        {
            return [];
        }
    }
}
