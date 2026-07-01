using Duck.Functional.Elementary;
using Duck.Functional.Value.Single;
using Duck.Functions.Value.Double;
using Duck.Functions.Value.Single;

namespace Duck.Modules.Basic
{
    public class SoftMax : IModule
    {
        private static readonly Apply<Exp> exp = new();
        private readonly SymmetricApplication<Max> max = new();
        private readonly SymmetricApplication<Add> sum = new();
        public Matrix Forward(Matrix m)
        {
            Matrix[] o = new Matrix[m.shape.width];
            for (int i = 0; i < m.shape.width; i++)
            {
                Matrix r = m[i];
                Matrix n = max.Apply(r);
                Matrix a = exp.Forward(r - n);
                Matrix s = sum.Apply(a);
                o[i] = a / s;
            }
            return new Concatenate(FunctionType.Row).Apply(o).T();
        }

        public Matrix[] GetParameters()
        {
            return [];
        }
    }
}
