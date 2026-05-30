using Duck.CustomLLM.Library.Objects.MatrixObjects;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Parameters
{
    public interface IParameter
    {
        public Matrix? result { get; set; }
        public Matrix[] MatricesUsed();
    }
}
