using Duck.CustomLLM.Library.Objects.MatrixObjects;
using Duck.Functions.Basic;
using Duck.Functions.Parameters;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Modules.Loss
{
    public interface ILoss<T> where T : IParameter
    {
        public Matrix Apply(T p);
    }
}
