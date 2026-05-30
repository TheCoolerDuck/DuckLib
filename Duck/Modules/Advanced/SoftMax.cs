using Duck.CustomLLM.Library.Objects.MatrixObjects;
using Duck.Functions.Basic;
using Duck.Functions.Value.Single;
using Duck.Modules.Basic;

namespace Duck.Modules.Advanced
{
    public class SoftMax(FunctionType type = FunctionType.Whole) : IModule
    {
        private static readonly Apply<Exp> exp = new();
        private readonly Mean mean = new(type);
        private readonly Sum sum = new(type);
        public Matrix Forward(Matrix m)
        {
            Matrix a = exp.Forward(m);
            return a / sum.Forward(a);
        }

        public Matrix[] GetParameters()
        {
            return [];
        }
    }
}
