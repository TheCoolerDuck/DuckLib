using Duck.Functions.Basic;
using Duck.Functions.Parameters;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Matrix_Utilities
{
    internal interface IBackwardContext
    {
        public void WalkBack();
        public void ZeroGradient();
    }

    internal class BackwardContext<T> : IBackwardContext where T : IParameter
    {
        private readonly IBasicFunction<T>? function; // null signifys loss function was aplied
        private readonly T parameters;

        public BackwardContext(IBasicFunction<T>? function, T parameters)
        {
            this.function = function;
            this.parameters = parameters;
            foreach (Matrix mat in parameters.MatricesUsed())
            {
                mat.matrixBase.usages.Add(this);
            }
        }
        public void WalkBack()
        {
            function?.ApplyGradient(parameters);

            foreach (Matrix mat in parameters.MatricesUsed())
            {
                mat.matrixBase.usages.Remove(this);
                mat.Backwards();
            }
        }

        public void ZeroGradient()
        {
            foreach (Matrix mat in parameters.MatricesUsed())
                mat.ZeroGradient();
        }

        public override string ToString()
        {
            return $"backwards contex - {function}, {parameters}";
        }
    }
}
