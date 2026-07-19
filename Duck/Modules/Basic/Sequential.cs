using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Modules.Basic
{
    public class Sequential(Module[] modules, Module? parent, string name = "Sequential") : Module(parent, name)
    {
        private readonly Module[] modules = modules;

        public override Matrix Forward(Matrix m)
        {
            foreach (Module module in modules)
                m = module.Forward(m);

            return m;
        }

        public override Matrix[] GetParameters()
        {
            List<Matrix> parameters = [];

            foreach (Module module in modules)
                parameters.AddRange(module.GetParameters());

            return [.. parameters];
        }
    }
}
