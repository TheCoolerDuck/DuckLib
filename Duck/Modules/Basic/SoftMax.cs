using Duck.Functional.Elementary;
using Duck.Functional.Elementary.SymmetricApplication;
using Duck.Functional.Elementary.Concatenate;
using Duck.Functional.Value.Single;
using Duck.Functions.Value.Double;
using Duck.Functions.Value.Single;

namespace Duck.Modules.Basic
{
    public class SoftMax : Module
    {
        private readonly Apply<Exp> exp;
        private readonly SymmetricApplication<Max> max = new();
        private readonly SymmetricApplication<Add> sum = new();

        public SoftMax(Module? parent, string name = "Softmax") : base(parent, name)
        {
            exp = new Apply<Exp>(this);
        }
        public override Matrix Forward(Matrix m)
        {
            Matrix[] o = new Matrix[m.shape.width];
            for (int i = 0; i < m.shape.width; i++)
            {
                Matrix r = m.GetRow(i);
                Matrix n = max.Apply(r);
                Matrix a = exp.Forward(r - n);
                Matrix s = sum.Apply(a);
                o[i] = a / s;
            }
            return new Concatenate(FunctionType.Row, m.shape.width).Apply(o).T();
        }

        public override Matrix[] GetParameters()
        {
            return [];
        }
    }
}
