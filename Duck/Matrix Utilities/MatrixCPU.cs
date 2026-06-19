using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Matrix_Utilities
{
    internal class MatrixCPU : MatrixBase
    {
        internal readonly float[,] values;
        internal float[,]? gradient;

        public new readonly (int width, int height) shape;
        internal MatrixCPU(float[,] values, bool hasGradient, IBackwardContext? backwardContext = null, string name = "") : base(backwardContext, name)
        {
            this.values = values;
            gradient = hasGradient ? new float[values.GetLength(0), values.GetLength(1)] : null;
            shape = (values.GetLength(0), values.GetLength(1));
        }

        protected override void ZeroGradientValues()
        {
            if (gradient == null)
                return;

            Array.Clear(gradient, 0, gradient.Length);
        }
        public override (int width, int height) GetShape()
        {
            return shape;
        }
        public override bool HasGradient()
        {
            return gradient != null;
        }

        #region Gets and Sets
        public float this[int x, int y, bool transposed]
        {
            get => Get(x, y, transposed);
            set => Set(x, y, value, transposed);
        }
        public float Get(int x, int y, bool transposed)
        {
            if (transposed)
                return values[y, x];
            return values[x, y];
        }
        public void Set(int x, int y, float value, bool transposed)
        {
            if (transposed)
                values[y, x] = value;
            else
                values[x, y] = value;
        }
        public float GetGradient(int x, int y, bool transposed)
        {
            if (gradient == null)
                throw new InvalidOperationException("Matrix does not have gradient");

            if (transposed)
                return gradient[y, x];
            return gradient[x, y];
        }
        public void AddGradient(int x, int y, float value, bool transposed)
        {
            if (gradient == null)
                return;

            lock (gradient)
            {
                if (transposed)
                    gradient[y, x] += value;
                else
                    gradient[x, y] += value;
            }
        }

        internal override float[,] GetValues()
        {
            float[,] output = new float[shape.width, shape.height];
            Array.Copy(values, output, values.Length);
            return output;
        }

        internal override float[,] GetGradients()
        {
            if (gradient == null)
                throw new ArgumentNullException();

            float[,] output = new float[shape.width, shape.height];
            Array.Copy(gradient, output, gradient.Length);
            return output;
        }
        #endregion
    }

}
