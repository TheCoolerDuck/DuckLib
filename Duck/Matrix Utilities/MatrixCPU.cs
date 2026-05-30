using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Matrix_Utilities
{
    internal class MatrixCPU : MatrixBase
    {
        internal readonly double[,] values;
        internal double[,]? gradient;

        public new readonly (int width, int height) shape;
        internal MatrixCPU(double[,] values, bool hasGradient, IBackwardContext? backwardContext = null, string name = "") : base(backwardContext, name)
        {
            this.values = values;
            gradient = hasGradient ? new double[values.GetLength(0), values.GetLength(1)] : null;
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
        public double this[int x, int y, bool transposed]
        {
            get => Get(x, y, transposed);
            set => Set(x, y, value, transposed);
        }
        public double Get(int x, int y, bool transposed)
        {
            if (transposed)
                return values[y, x];
            return values[x, y];
        }
        public void Set(int x, int y, double value, bool transposed)
        {
            if (transposed)
                values[y, x] = value;
            else
                values[x, y] = value;
        }
        public double GetGradient(int x, int y, bool transposed)
        {
            if (gradient == null)
                throw new InvalidOperationException("Matrix does not have gradient");

            if (transposed)
                return gradient[y, x];
            return gradient[x, y];
        }
        public void AddGradient(int x, int y, double value, bool transposed)
        {
            if (gradient == null)
                return;

            if (transposed)
                gradient[y, x] += value;
            else
                gradient[x, y] += value;
        }
        #endregion

        #region ToString

        private const int toStringMaxSize = 3;

        public override string ToString()
        {
            return ToString(false);
        }

        internal override string ToString(bool transposed)
        {
            StringBuilder sb = new();

            sb.AppendLine($"Matrix {name}-{ID:X}; Shape: " + (transposed ? (shape.height, shape.width) : shape));

            sb.AppendLine("Values:");
            AddMatrix(Get);

            if (gradient != null)
            {
                sb.AppendLine("Gradient:");
                AddMatrix(GetGradient);
            }

            return sb.ToString();

            void AddMatrix(Func<int, int, bool, double> func)
            {
                sb.Append("[ ");

                int width = transposed ? shape.height : shape.width;
                int height = transposed ? shape.width : shape.height;

                for (int x = 0; x < (width > toStringMaxSize ? toStringMaxSize : width); x++)
                {
                    for (int y = 0; y < (height > toStringMaxSize ? toStringMaxSize : height); y++)
                    {
                        bool xDots = x == toStringMaxSize - 2 && width > toStringMaxSize;
                        bool yDots = y == toStringMaxSize - 2 && height > toStringMaxSize;

                        if (xDots && yDots) sb.Append("⋱  ");
                        else if (yDots) sb.Append("⋯  ");
                        else if (xDots) sb.Append("   ⋮   ");
                        else
                        {
                            bool xEnd = x == toStringMaxSize - 1;
                            bool yEnd = y == toStringMaxSize - 1;

                            if (xEnd && yEnd) sb.Append(FormatDouble(func(width - 1, height - 1, transposed)) + ", ");
                            else if (xEnd) sb.Append(FormatDouble(func(width - 1, y, transposed)) + ", ");
                            else if (yEnd) sb.Append(FormatDouble(func(x, height - 1, transposed)) + ", ");
                            else sb.Append(FormatDouble(func(x, y, transposed)) + ", ");
                        }
                    }

                    sb.Append("\n  ");
                }

                sb.Remove(sb.Length - 5, 5);

                sb.Append(" ]\n");
            }

            static string FormatDouble(double f)
            {
                if (double.IsNaN(f))
                    return " Nan ";

                if (f >= 0)
                {
                    if (f == double.PositiveInfinity) return "  +∞ ";
                    if (f >= 10_000) return "#####";
                    if (f >= 1_000) return ((int)f).ToString();
                    if (f >= 100) return f.ToString("F1");
                    if (f >= 10) return f.ToString("F2");
                    return Math.Abs(f).ToString("F3");
                }
                else
                {
                    if (f == double.NegativeInfinity) return "  -∞ ";
                    if (f <= -1_000) return "-####";
                    if (f <= -100) return ((int)f).ToString();
                    if (f <= -10) return f.ToString("F1");
                    return f.ToString("F2");
                }
            }
        }

        #endregion
    }

}
