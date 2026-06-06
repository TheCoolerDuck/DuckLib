using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck
{
    using Device_Management;
    using Duck.Functions.Basic;
    using Duck.Functions.Value.Double;
    using Duck.Matrix_Utilities;
    using ManagedCuda;
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Xml.Linq;
    using ManagedCuda.NVRTC;

    namespace CustomLLM.Library.Objects.MatrixObjects
    {
        public class Matrix
        {

            #region Innit
            public Matrix(float[,] values) : this(values, true, null, DeviceManager.defaultDevice) { }
            public Matrix(float[,] values, string name) : this(values, true, null, DeviceManager.defaultDevice, name) { }

            public Matrix(float[,] values, bool hasGrad) : this(values, hasGrad, null, DeviceManager.defaultDevice) { }
            public Matrix(float[,] values, bool hasGrad, string name) : this(values, hasGrad, null, DeviceManager.defaultDevice, name) { }

            public Matrix(float[,] values, Device device) : this(values, true, null, device) { }
            public Matrix(float[,] values, Device device, string name) : this(values, true, null, device, name) { }

            public Matrix(float[,] values, bool hasGrad, Device device) : this(values, hasGrad, null, device) { }
            public Matrix(float[,] values, bool hasGrad, Device device, string name) : this(values, hasGrad, null, device, name) { }

            internal Matrix(float[,] values, IBackwardContext backwardContext) : this(values, true, backwardContext, DeviceManager.defaultDevice) { }
            internal Matrix(float[,] values, IBackwardContext backwardContext, string name) : this(values, true, backwardContext, DeviceManager.defaultDevice, name) { }

            internal Matrix(float[,] values, bool hasGrad, IBackwardContext backwardContext) : this(values, hasGrad, backwardContext, DeviceManager.defaultDevice) { }
            internal Matrix(float[,] values, bool hasGrad, IBackwardContext backwardContext, string name) : this(values, hasGrad, backwardContext, DeviceManager.defaultDevice, name) { }

            internal Matrix(float[,] values, IBackwardContext backwardContext, Device device) : this(values, true, backwardContext, device) { }
            internal Matrix(float[,] values, IBackwardContext backwardContext, Device device, string name) : this(values, true, backwardContext, device, name) { }
            internal Matrix(float[,] values, bool hasGrad, IBackwardContext? backwardContext, Device device, string name = "")
            {
                this.device = device;
                if (device == Device.CPU)
                    matrixBase = new MatrixCPU(values, hasGrad, backwardContext, name);
                else
                    matrixBase = new MatrixGPU(values, backwardContext, name);
            }
            internal Matrix((int width, int height) shape, CudaDeviceVariable<float> values, IBackwardContext backwardContext)
            {
                matrixBase = new MatrixGPU(shape, values, backwardContext);
            }
            internal Matrix(Matrix m)
            {
                matrixBase = m.matrixBase;
                transposed = !m.transposed;
            }

            #endregion
            public readonly Device device;
            internal readonly MatrixBase matrixBase;


            #region Shape

            public readonly bool transposed = false;
            public Matrix T()
            {
                return new Matrix(this);
            }

            public (int width, int height) shape => GetShape();

            public (int width, int height) GetShape()
            {
                (int width, int height) = matrixBase.shape;
                if (transposed)
                    return (height, width);
                return (width, height);
            }

            public bool IsScalar()
            {
                return shape.Equals((1, 1));
            }

            public bool IsVector()
            {
                return IsRowVector() || IsColVector();
            }
            public bool IsRowVector()
            {
                return !IsScalar() && shape.height == 1;
            }
            public bool IsColVector()
            {
                return !IsScalar() && shape.width == 1;
            }

            #endregion

            public float[,] values => matrixBase.GetValues();
            public float[,] gradients => matrixBase.GetGradients();

            public void Backwards() => matrixBase.Backwards();

            public void ZeroGradient() => matrixBase.ZeroGradient();

            internal GPUMatrixStruct GPUValues()
            {
                return new GPUMatrixStruct(this, ((MatrixGPU)matrixBase).values);
            }
            internal GPUMatrixStruct GPUGradient()
            {
                return new GPUMatrixStruct(this, ((MatrixGPU)matrixBase).gradient!);
            }


            #region ToString
            private const int toStringMaxSize = 8;
            public override string ToString()
            {
                StringBuilder sb = new();

                sb.AppendLine($"Matrix {matrixBase.name}; ID: {matrixBase.ID:X}; Shape: " + (transposed ? (shape.height, shape.width) : shape));

                sb.AppendLine("Values:");

                float[,] values = matrixBase.GetValues();
                AddMatrix((x, y, t) =>  t ? values[y, x] : values[x, y]);

                if (matrixBase.HasGradient())
                {
                    sb.AppendLine("Gradient:");
                    float[,] gradients = matrixBase.GetGradients();
                    AddMatrix((x, y, t) => t ? gradients[y, x] : gradients[x, y]);
                }

                return sb.ToString();

                void AddMatrix(Func<int, int, bool, float> func)
                {
                    sb.Append("[ ");

                    for (int y = 0; y < (shape.height > toStringMaxSize ? toStringMaxSize : shape.height); y++)
                        
                    {
                        for (int x = 0; x < (shape.width > toStringMaxSize ? toStringMaxSize : shape.width); x++)
                        {
                            bool xDots = x == toStringMaxSize - 2 && shape.width > toStringMaxSize;
                            bool yDots = y == toStringMaxSize - 2 && shape.height > toStringMaxSize;

                            if (xDots && yDots) sb.Append("⋱  ");
                            else if (yDots) sb.Append("⋯  ");
                            else if (xDots) sb.Append("   ⋮   ");
                            else
                            {
                                bool xEnd = x == toStringMaxSize - 1;
                                bool yEnd = y == toStringMaxSize - 1;

                                if (xEnd && yEnd) sb.Append(Formatfloat(func(shape.width - 1, shape.height - 1, transposed)) + ", ");
                                else if (xEnd) sb.Append(Formatfloat(func(shape.width - 1, y, transposed)) + ", ");
                                else if (yEnd) sb.Append(Formatfloat(func(x, shape.height - 1, transposed)) + ", ");
                                else sb.Append(Formatfloat(func(x, y, transposed)) + ", ");
                            }
                        }

                        sb.Append("\n  ");
                    }

                    sb.Remove(sb.Length - 5, 5);

                    sb.Append(" ]\n");
                }

                static string Formatfloat(float f)
                {
                    if (float.IsNaN(f))
                        return " Nan ";

                    if (f >= 0)
                    {
                        if (f == float.PositiveInfinity) return "  +∞ ";
                        if (f >= 10_000) return "#####";
                        if (f >= 1_000) return ((int)f).ToString();
                        if (f >= 100) return f.ToString("F1");
                        if (f >= 10) return f.ToString("F2");
                        return Math.Abs(f).ToString("F3");
                    }
                    else
                    {
                        if (f == float.NegativeInfinity) return "  -∞ ";
                        if (f <= -1_000) return "-####";
                        if (f <= -100) return ((int)f).ToString();
                        if (f <= -10) return f.ToString("F1");
                        return f.ToString("F2");
                    }
                }
            }

            #endregion



            #region Operations

            #region Function Objects
            private static readonly MatrixMultiplication matMulObj = new();
            private static readonly MatrixFunction<Div> divObj = new();
            private static readonly MatrixFunction<Sub> subObj = new();
            private static readonly MatrixFunction<Add> addObj = new();
            private static readonly MatrixFunction<Mul> mulObj = new();
            private static readonly GetVectors getRowObj = new(FunctionType.Row);
            private static readonly GetVectors getColObj = new(FunctionType.Column);
            private static readonly Concatenate rowCatObj = new(FunctionType.Row);
            private static readonly Concatenate colCatObj = new(FunctionType.Column);

            #endregion



            #region Functions
            public Matrix matMul(Matrix o) { return matMulObj.Apply((this, o)); }
            public Matrix div(Matrix o) { return divObj.Apply((this, o)); }
            public Matrix sub(Matrix o) { return subObj.Apply((this, o)); }
            public Matrix add(Matrix o) { return addObj.Apply((this, o)); }
            public Matrix mul(Matrix o) { return mulObj.Apply((this, o)); }
            public Matrix div(float o) { return divObj.Apply((this, new Matrix(new float[,] { { o } }, false, device))); }
            public Matrix iDiv(float o) { return divObj.Apply((new Matrix(new float[,] { { o } }, false, device), this)); }
            public Matrix sub(float o) { return subObj.Apply((this, new Matrix(new float[,] { { o } }, false, device))); }
            public Matrix iSub(float o) { return subObj.Apply((new Matrix(new float[,] { { o } }, false, device), this)); }
            public Matrix add(float o) { return addObj.Apply((this, new Matrix(new float[,] { { o } }, false, device))); }
            public Matrix mul(float o) { return mulObj.Apply((this, new Matrix(new float[,] { { o } }, false, device))); }
            public Matrix getRow(int i) { return getRowObj.Apply((this, [i])); }
            public Matrix getCol(int i) { return getColObj.Apply((this, [i])); }
            public Matrix getRows(int[] i) { return getRowObj.Apply((this, i)); }
            public Matrix getCosl(int[] i) { return getColObj.Apply((this, i)); }
            public Matrix rowConcatenate(Matrix o) { return rowCatObj.Apply(new Matrix[] { this, o }); }
            public Matrix colConcatenate(Matrix o) { return colCatObj.Apply(new Matrix[] { this, o }); }

            #endregion

            #region Operators

            public static Matrix operator <<(Matrix a, Matrix b) { return a.matMul(b); }
            public static Matrix operator /(Matrix a, Matrix b) { return a.div(b); }
            public static Matrix operator -(Matrix a, Matrix b) { return a.sub(b); }
            public static Matrix operator +(Matrix a, Matrix b) { return a.add(b); }
            public static Matrix operator *(Matrix a, Matrix b) { return a.mul(b); }
            public static Matrix operator /(float a, Matrix b) { return b.iDiv(a); }
            public static Matrix operator -(float a, Matrix b) { return b.iSub(a); }
            public static Matrix operator +(float a, Matrix b) { return b.add(a); }
            public static Matrix operator *(float a, Matrix b) { return b.mul(a); }
            public static Matrix operator /(Matrix a, float b) { return a.div(b); }
            public static Matrix operator -(Matrix a, float b) { return a.sub(b); }
            public static Matrix operator +(Matrix a, float b) { return a.add(b); }
            public static Matrix operator *(Matrix a, float b) { return a.mul(b); }
            public static Matrix operator -(Matrix a) { return a.mul(-1); }
            public Matrix this[int i, FunctionType type]
            {
                get
                {
                    if (type == FunctionType.Row)
                        return getRow(i);
                    if (type == FunctionType.Column)
                        return getCol(i);

                    throw new ArgumentException("FunctionType.whole is not allowed.", nameof(type));
                }
            }
            public Matrix this[int i]
            {
                get
                {
                    return getRow(i);
                }
            }
            public Matrix this[int i, char c]
            {
                get
                {
                    return getRow(i);
                }
            }
            public Matrix this[char c, int i]
            {
                get
                {
                    return getCol(i);
                }
            }
            public static Matrix operator &(Matrix a, Matrix b) { return a.rowConcatenate(b); }
            public static Matrix operator |(Matrix a, Matrix b) { return a.colConcatenate(b); }
            #endregion
            #endregion



            #region Generators
            public static float[,] Random(int width, int height)
            {
                return Random(width, height, 1);
            }

            public static float[,] Random(int width, int height, float denominator)
            {
                float[,] values = new float[width, height];

                Random rand = new();

                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < height; y++)
                    {
                        values[x, y] = (float)((rand.NextDouble() - 0.5) * 2 / denominator);
                    }
                }

                return values;
            }

            public static float[,] Zeros(int width, int height)
            {
                float[,] values = new float[width, height];

                return values;
            }

            public static float[,] Ones(int width, int height)
            {
                float[,] values = new float[width, height];

                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < height; y++)
                    {
                        values[x, y] = 1;
                    }
                }

                return values;
            }

            public static float[,] OneHot(int value, int categories, int width = 1)
            {
                float[,] values = new float[width, categories];

                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < categories; y++)
                    {
                        values[0, y] = y == value ? 1 : 0;
                    }
                }

                return values;
            }



            #endregion
        }
    }

}
