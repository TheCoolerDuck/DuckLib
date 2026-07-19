using Duck.Functions.Value.Single;
using Duck.Modules.Basic;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Modules.Normalization
{
    public class LinearTanh : Module
    {
        private readonly Linear l;
        private readonly Apply<TanH> tanh;

        public LinearTanh(int vectorSize, Module? parent, string name = "LinearTanhNorm") : base(parent, name)
        {
            l = new(vectorSize, vectorSize, this);
            tanh = new(this);
        }
        public override Matrix Forward(Matrix m)
        {
            return tanh.Forward(l.Forward(m));
        }

        public override Matrix[] GetParameters()
        {
            return [..l.GetParameters()];
        }
    }
}
