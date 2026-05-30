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
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;

    namespace CustomLLM.Library.Objects.MatrixObjects
    {
        public class Matrix
        {

            #region Innit
            public Matrix(double[,] values) : this(values, true, null, DeviceManager.defaultDevice, null) { }
            public Matrix(double[,] values, string name) : this(values, true, null, DeviceManager.defaultDevice, name) { }

            public Matrix(double[,] values, bool hasGrad) : this(values, hasGrad, null, DeviceManager.defaultDevice, null) { }
            public Matrix(double[,] values, bool hasGrad, string name) : this(values, hasGrad, null, DeviceManager.defaultDevice, name) { }

            public Matrix(double[,] values, Device device) : this(values, true, null, device, null) { }
            public Matrix(double[,] values, Device device, string name) : this(values, true, null, device, name) { }

            public Matrix(double[,] values, bool hasGrad, Device device) : this(values, hasGrad, null, device, null) { }
            public Matrix(double[,] values, bool hasGrad, Device device, string name) : this(values, hasGrad, null, device, name) { }

            internal Matrix(double[,] values, IBackwardContext? backwardContext) : this(values, true, backwardContext, DeviceManager.defaultDevice, null) { }
            internal Matrix(double[,] values, IBackwardContext? backwardContext, string name) : this(values, true, backwardContext, DeviceManager.defaultDevice, name) { }

            internal Matrix(double[,] values, bool hasGrad, IBackwardContext? backwardContext) : this(values, hasGrad, backwardContext, DeviceManager.defaultDevice, null) { }
            internal Matrix(double[,] values, bool hasGrad, IBackwardContext? backwardContext, string name) : this(values, hasGrad, backwardContext, DeviceManager.defaultDevice, name) { }

            internal Matrix(double[,] values, IBackwardContext? backwardContext, Device device) : this(values, true, backwardContext, device, null) { }
            internal Matrix(double[,] values, IBackwardContext? backwardContext, Device device, string name) : this(values, true, backwardContext, device, name) { }

            internal Matrix(double[,] values, bool hasGrad, IBackwardContext? backwardContext, Device device) : this(values, hasGrad, backwardContext, device, null) { }
            internal Matrix(double[,] values, bool hasGrad, IBackwardContext? backwardContext, Device device, string name = "")
            {
                this.device = device;
                if (device == Device.CPU)
                    matrixBase = new MatrixCPU(values, hasGrad, backwardContext, name);
                else
                    throw new NotImplementedException();
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



            public void Backwards() => matrixBase.Backwards();

            public void ZeroGradient() => matrixBase.ZeroGradient();


            #region ToString

            public override string ToString()
            {
                return matrixBase.ToString(transposed)!;
            }

            #endregion



            #region Operations

            #region Function Objects
            private static readonly MatrixMultiplication matMulObj = new();
            private static readonly MatrixFunction<Div> divObj = new();
            private static readonly MatrixFunction<Sub> subObj = new();
            private static readonly MatrixFunction<Add> addObj = new();
            private static readonly MatrixFunction<Mul> mulObj = new();
            private static readonly GetVector getRowObj = new(FunctionType.Row);
            private static readonly GetVector getColObj = new(FunctionType.Column);
            private static readonly Concatenate rowCatObj = new(FunctionType.Row);
            private static readonly Concatenate colCatObj = new(FunctionType.Column);

            #endregion



            #region Functions
            public Matrix matMul(Matrix o) { return matMulObj.Apply((this, o)); }
            public Matrix div(Matrix o) { return divObj.Apply((this, o)); }
            public Matrix sub(Matrix o) { return subObj.Apply((this, o)); }
            public Matrix add(Matrix o) { return addObj.Apply((this, o)); }
            public Matrix mul(Matrix o) { return mulObj.Apply((this, o)); }
            public Matrix div(double o) { return divObj.Apply((this, new Matrix(new double[,] { { o } }, false, device))); }
            public Matrix iDiv(double o) { return divObj.Apply((new Matrix(new double[,] { { o } }, false, device), this)); }
            public Matrix sub(double o) { return subObj.Apply((this, new Matrix(new double[,] { { o } }, false, device))); }
            public Matrix iSub(double o) { return subObj.Apply((new Matrix(new double[,] { { o } }, false, device), this)); }
            public Matrix add(double o) { return addObj.Apply((this, new Matrix(new double[,] { { o } }, false, device))); }
            public Matrix mul(double o) { return mulObj.Apply((this, new Matrix(new double[,] { { o } }, false, device))); }
            public Matrix getRow(int i) { return getRowObj.Apply((this, i)); }
            public Matrix getCol(int i) { return getColObj.Apply((this, i)); }
            public Matrix rowConcatenate(Matrix o) { return rowCatObj.Apply(new Matrix[] { this, o }); }
            public Matrix colConcatenate(Matrix o) { return colCatObj.Apply(new Matrix[] { this, o }); }

            #endregion

            #region Operators

            public static Matrix operator <<(Matrix a, Matrix b) { return a.matMul(b); }
            public static Matrix operator /(Matrix a, Matrix b) { return a.div(b); }
            public static Matrix operator -(Matrix a, Matrix b) { return a.sub(b); }
            public static Matrix operator +(Matrix a, Matrix b) { return a.add(b); }
            public static Matrix operator *(Matrix a, Matrix b) { return a.mul(b); }
            public static Matrix operator /(double a, Matrix b) { return b.iDiv(a); }
            public static Matrix operator -(double a, Matrix b) { return b.iSub(a); }
            public static Matrix operator +(double a, Matrix b) { return b.add(a); }
            public static Matrix operator *(double a, Matrix b) { return b.mul(a); }
            public static Matrix operator /(Matrix a, double b) { return a.div(b); }
            public static Matrix operator -(Matrix a, double b) { return a.sub(b); }
            public static Matrix operator +(Matrix a, double b) { return a.add(b); }
            public static Matrix operator *(Matrix a, double b) { return a.mul(b); }
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
            public static double[,] Random(int width, int height)
            {
                return Random(width, height, 1);
            }

            public static double[,] Random(int width, int height, double denominator)
            {
                double[,] values = new double[width, height];

                Random rand = new();

                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < height; y++)
                    {
                        values[x, y] = (rand.NextDouble() - 0.5) * 2 / denominator;
                    }
                }

                return values;
            }

            public static double[,] Zeros(int width, int height)
            {
                double[,] values = new double[width, height];

                return values;
            }

            public static double[,] Ones(int width, int height)
            {
                double[,] values = new double[width, height];

                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < height; y++)
                    {
                        values[x, y] = 1;
                    }
                }

                return values;
            }

            public static double[,] OneHot(int value, int categories, int width = 1)
            {
                double[,] values = new double[width, categories];

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
