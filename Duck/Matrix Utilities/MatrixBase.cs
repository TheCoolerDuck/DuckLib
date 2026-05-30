using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Matrix_Utilities
{
    internal abstract class MatrixBase
    {
        private static int IDn = 0;
        public readonly int ID = IDn++;
        public readonly string name;
        internal IBackwardContext? backwardContext;
        public (int width, int height) shape => GetShape();
        public bool hasGradient => HasGradient();
        protected bool gradientIsDirty = false;
        internal List<IBackwardContext> usages = [];
        internal object? optimizerData;
        internal MatrixBase(IBackwardContext? backwardContext = null, string name = "")
        {
            this.backwardContext = backwardContext;
            this.name = name;
        }

        public void Backwards()
        {
            if (usages.Count == 0)
            {
                gradientIsDirty = true;
                backwardContext?.WalkBack();
            }
        }
        public void ZeroGradient()
        {
            if ((hasGradient && gradientIsDirty) || backwardContext != null)
            {
                ZeroGradientValues();
                backwardContext?.ZeroGradient();
                backwardContext = null;
                usages.Clear();
                gradientIsDirty = false;
            }
        }
        public abstract (int width, int height) GetShape();
        public abstract bool HasGradient();
        protected abstract void ZeroGradientValues();
        internal abstract string ToString(bool transposed);
    }
}
