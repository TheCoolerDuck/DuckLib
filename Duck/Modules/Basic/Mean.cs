using Duck.CustomLLM.Library.Objects.MatrixObjects;
using Duck.Functions.Basic;
using Duck.Modules;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Modules.Basic
{
    public class Mean(FunctionType type) : IModule
    {
        private readonly FunctionType type = type;
        private readonly BSum sum = new(type);

        public Matrix Forward(Matrix m)
        {
            int denominator = type == FunctionType.Row ? m.shape.width : type == FunctionType.Column ? m.shape.height : m.shape.width * m.shape.height;
            return sum.Apply(m) / denominator;
        }

        public Matrix[] GetParameters()
        {
            return [];
        }
    }
}
