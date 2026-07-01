using Duck.Functions.Value.Single;
using Duck.Modules.Basic;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Modules.Normalization
{
    public class LinearTanh(int vectorSize) : IModule
    {
        private Linear l = new(vectorSize, vectorSize, "norm linear");
        private Apply<TanH> tanh = new();
        public Matrix Forward(Matrix m)
        {
            return tanh.Forward(l.Forward(m));
        }

        public Matrix[] GetParameters()
        {
            return [..l.GetParameters()];
        }
    }
}
