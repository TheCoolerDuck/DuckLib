using Duck.Functions.Parameters;
using Duck.Management;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Matrix_Utilities
{
    internal abstract class MatrixBase
    {
        public Device device => this is MatrixCPU ? Device.CPU : Device.GPU;
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
                if (hasGradient)
                {
                    gradientIsDirty = true;
                    backwardContext?.WalkBack();
                }
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

        public void Detach(MatrixBase? parent)
        {
            if (usages.Count <= 0 && parent != null)
                return;

            if (parent == null)
                backwardContext = null;

            if (usages.Count <= 1)
            {

                usages.Clear();

                if (backwardContext != null)
                {
                    foreach (Matrix matrix in backwardContext.parameter.MatricesUsed())
                        matrix.matrixBase.Detach(this);
                }

                return;
            }

            usages.RemoveAll(u => u.parameter.result!.matrixBase == parent);
        }
        public abstract (int width, int height) GetShape();
        public abstract bool HasGradient();
        protected abstract void ZeroGradientValues();
        internal abstract float[,] GetValues();
        internal abstract float[,] GetGradients();

        public string InterfaceString(int resultSize, bool transposed)
        {
            StringBuilder sb = new();

            sb.AppendLine(
                $"Matrix {name}; ID: {ID}; Shape: {shape}"
            );

            sb.AppendLine("Values:");

            float[,] values = GetValues();

            AddMatrix((x, y) => transposed ? values[y, x] : values[x, y]);

            if (HasGradient())
            {
                sb.AppendLine("Gradient:");

                float[,] gradients = GetGradients();

                AddMatrix((x, y) => transposed ? gradients[y, x] : gradients[x, y]);
            }

            void AddMatrix(Func<int, int, float> func)
            {
                int w = shape.width;
                int h = shape.height;

                int maxX = Math.Min(resultSize, w);
                int maxY = Math.Min(resultSize, h);

                sb.Append("[ ");

                for (int y = 0; y < maxY; y++)
                {
                    if (y > 0)
                        sb.Append("  ");

                    for (int x = 0; x < maxX; x++)
                    {
                        bool xDots = x == resultSize - 2 && w > resultSize;
                        bool yDots = y == resultSize - 2 && h > resultSize;

                        if (xDots && yDots)
                        {
                            sb.Append("⋱  ");
                        }
                        else if (xDots)
                        {
                            sb.Append("⋯  ");
                        }
                        else if (yDots)
                        {
                            sb.Append("  ⋮  ");
                        }
                        else
                        {
                            sb.Append(FormatFloat(func(x, y)));
                        }

                        if (x < maxX - 1)
                            sb.Append(", ");
                    }

                    if (y < maxY - 1)
                        sb.AppendLine();
                }

                sb.Append(" ]\n");
            }

            return sb.ToString();
        }

        public string DataString(string format, bool transposed)
        {
            StringBuilder sb = new();

            sb.Append($"{name}:{shape.width},{shape.height}:{device}:{format}:");

            float[,] values = GetValues();

            byte[] data = new byte[shape.width * shape.height * sizeof(uint)];
            int offset = 0;

            for (int x = 0; x < shape.width; x++)
            {
                for (int y = 0; y < shape.height; y++)
                {
                    uint bits = BitConverter.SingleToUInt32Bits(
                        transposed ? values[y, x] : values[x, y]);

                    BitConverter.GetBytes(bits)
                        .CopyTo(data, offset);

                    offset += sizeof(uint);
                }
            }

            sb.Append(Convert.ToBase64String(data));

            return sb.ToString();
        }

        private static string FormatFloat(float f)
        {
            if (float.IsNaN(f)) return " Nan ";
            if (f == float.PositiveInfinity) return " +∞ ";
            if (f == float.NegativeInfinity) return " -∞ ";

            if (f >= 0)
            {
                if (f >= 10_000) return "####";
                if (f >= 1_000) return ((int)f).ToString();
                if (f >= 100) return f.ToString("F1");
                if (f >= 10) return f.ToString("F2");

                return f.ToString("F3");
            }
            else
            {
                if (f <= -1_000) return "-###";
                if (f <= -100) return ((int)f).ToString();
                if (f <= -10) return f.ToString("F1");

                return f.ToString("F2");
            }
        }
    }
}
