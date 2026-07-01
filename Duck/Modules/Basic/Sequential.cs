using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Modules.Basic
{
    public class Sequential(IModule[] modules) : IModule
    {
        private readonly IModule[] modules = modules;

        public Matrix Forward(Matrix m)
        {
            foreach (IModule module in modules)
                m = module.Forward(m);

            return m;
        }

        public Matrix[] GetParameters()
        {
            List<Matrix> parameters = [];

            foreach (IModule module in modules)
                parameters.AddRange(module.GetParameters());

            return [.. parameters];
        }

        public static Sequential MLP(int inSize, int[] hiddenSizes, int outSize, IModule activation, string name = "MLP")
        {
            List<IModule> modules = [];

            int i = 0;
            foreach (int h in hiddenSizes)
            {
                modules.Add(new Linear(inSize, h, $"{name}-linear {i++}"));
                modules.Add(activation);
                inSize = h;
            }

            modules.Add(new Linear(inSize, outSize, $"{name}-linear {i++}"));

            return new Sequential([.. modules]);
        }
    }
}
