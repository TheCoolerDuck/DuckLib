using Duck.Functional.Elementary;
using Duck.Functional.Elementary.Concatenate;
using Duck.Functional.Elementary.MatrixFunction;
using Duck.Functional.Elementary.Extend;
using Duck.Functional.Elementary.MatrixMultiplication;
using Duck.Functional.Elementary.GetVectors;
using Duck.Functions.Value.Double;
using Duck.Management;
using Duck.Matrix_Utilities;
using ManagedCuda;
using System.Globalization;
using System.Text;

namespace Duck
{
    public struct MatrixOptions
    {
        public MatrixOptions() { }
        public bool HasGrad { get; set; } = true;
        public Device Device { get; set; } = DeviceManager.defaultDevice;
        public string Name { get; set; } = "matrix";
    }

    public class Matrix : IFormattable
    {
        #region Fields

        internal readonly MatrixBase matrixBase;
        public readonly bool transposed = false;

        public Device device => matrixBase.device;

        public bool hasGradient => matrixBase.hasGradient;

        public int ID => matrixBase.ID;

        #endregion

        #region Initialization

        public Matrix(float[,] values, MatrixOptions options)
            : this(values, options, null) { }

        public Matrix(float[,] values)
            : this(values, new MatrixOptions(), null) { }

        public Matrix(ValuePackage values, MatrixOptions options)
           : this(values.values, options, null) { }

        public Matrix(ValuePackage values)
            : this(values.values, new MatrixOptions(), null) { }

        internal Matrix(float[,] values, IBackwardContext? backwardContext)
            : this(values, new MatrixOptions(), backwardContext) { }

        internal Matrix(float[,] values, MatrixOptions options, IBackwardContext? backwardContext)
        {
            matrixBase =
                options.Device == Device.CPU
                    ? new MatrixCPU(values, options.HasGrad, backwardContext, options.Name)
                    : new MatrixGPU(values, backwardContext, options.HasGrad, options.Name);
        }

        internal Matrix((int width, int height) shape, CudaDeviceVariable<float> values, IBackwardContext ctx, bool hasGradient = true)
        {
            matrixBase = new MatrixGPU(shape, values, ctx, hasGradient);
        }

        public Matrix(string dataString, bool hasGrad = true)
        {
            string[] items = dataString.Split(':');

            string name = items[0];

            string[] sShape = items[1].Split(",");
            int width = int.Parse(sShape[0]);
            int height = int.Parse(sShape[1]);

            string sDevice = items[2];
            Device device = Enum.Parse<Device>(sDevice);

            string format = items[3];

            string sValues = items[4];

            byte[] bValues = format switch
            {
                "B64" => Convert.FromBase64String(sValues),
                _ => throw new FormatException($"Unrecognized format: '{format}'")
            };

            uint[] iValues = new uint[bValues.Length / sizeof(uint)];
            Buffer.BlockCopy(bValues, 0, iValues, 0, bValues.Length);

            float[] fValues = [.. iValues.Select(i => BitConverter.UInt32BitsToSingle(i))];

            float[,] values = new float[width, height];

            Buffer.BlockCopy(fValues, 0, values, 0, fValues.Length * sizeof(float));

            matrixBase =
                device == Device.CPU
                    ? new MatrixCPU(values, hasGrad, null, name)
                    : new MatrixGPU(values, null, hasGrad, name);
        }
        internal Matrix(MatrixBase matrixBase)
        {
            this.matrixBase = matrixBase;
        }

        private Matrix(Matrix m)
        {
            matrixBase = m.matrixBase;
            transposed = !m.transposed;
        }

        #endregion

        #region Shape

        public int size => shape.width * shape.height;
        public (int width, int height) shape => GetShape();

        public (int width, int height) GetShape()
        {
            var (w, h) = matrixBase.shape;
            return transposed ? (h, w) : (w, h);
        }

        public bool IsScalar() => shape == (1, 1);
        public bool IsVector() => IsRowVector() || IsColVector();
        public bool IsRowVector() => !IsScalar() && shape.height == 1;
        public bool IsColVector() => !IsScalar() && shape.width == 1;
        public bool Is2D() => !IsVector() && !IsScalar();

        public Matrix T() => new(this);

        #endregion

        #region Data Access

        public float[,] values
        {
            get
            {
                float[,] src = matrixBase.GetValues();

                if (!transposed)
                    return src;

                int w = src.GetLength(0);
                int h = src.GetLength(1);

                float[,] t = new float[h, w];

                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        t[y, x] = src[x, y];
                    }
                }

                return t;
            }
        }
        public float[,] gradients
        {
            get
            {
                float[,] src = matrixBase.GetGradients();

                if (!transposed)
                    return src;

                int w = src.GetLength(0);
                int h = src.GetLength(1);

                float[,] t = new float[h, w];

                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        t[y, x] = src[x, y];
                    }
                }

                return t;
            }
        }
        public void Backwards() => matrixBase.Backwards();
        public void ZeroGradient() => matrixBase.ZeroGradient();

        internal GPUMatrixStruct GPUValues()
            => new(this, ((MatrixGPU)matrixBase).values);

        internal GPUMatrixStruct GPUGradient()
            => new(this, ((MatrixGPU)matrixBase).gradient!);

        public (int x, int y) XYFromIndex(int i)
        {
            return (i % shape.width, i / shape.width);
        }

        public int IndexFromXY(int x, int y)
        {
            return y * shape.width + x;
        }

        public Matrix Clone() => new(matrixBase.Clone());
        public Matrix CloneValues() => new(matrixBase.CloneValues());

        #endregion

        #region Formatting

        private const int DefaultSize = 5;

        public override string ToString()
            => ToString(DefaultSize.ToString(), CultureInfo.CurrentCulture);

        public string ToString(string? format, IFormatProvider? provider)
        {
            format ??= DefaultSize.ToString();

            return format switch
            {
                "B64" => DataString(format),
                _ when int.TryParse(format, out int size) => InterfaceString(size),
                _ => throw new FormatException($"Unrecognized format: '{format}'")
            };
        }
        public string InterfaceString(int resultSize) => matrixBase.InterfaceString(resultSize, transposed);
        public string DataString(string format) => matrixBase.DataString(format, transposed);

        #endregion


        #region Operations

        public Matrix MatMul(Matrix o) => new MatrixMultiplication().Apply((this, o));
        public Matrix Div(Matrix o) => new MatrixFunction<Div>().Apply(Broadcast(this, o));
        public Matrix Sub(Matrix o) => new MatrixFunction<Sub>().Apply(Broadcast(this, o));
        public Matrix Add(Matrix o) => new MatrixFunction<Add>().Apply(Broadcast(this, o));
        public Matrix Mul(Matrix o) => new MatrixFunction<Mul>().Apply(Broadcast(this, o));

        public Matrix Div(float o) => new MatrixFunction<Div>().Apply(Broadcast(this, Scalar(o)));
        public Matrix InvDiv(float o) => new MatrixFunction<Div>().Apply(Broadcast(Scalar(o), this));

        public Matrix Sub(float o) => new MatrixFunction<Sub>().Apply(Broadcast(this, Scalar(o)));
        public Matrix InvSub(float o) => new MatrixFunction<Sub>().Apply(Broadcast(Scalar(o), this));

        public Matrix Add(float o) => new MatrixFunction<Add>().Apply(Broadcast(this, Scalar(o)));
        public Matrix Mul(float o) => new MatrixFunction<Mul>().Apply(Broadcast(this, Scalar(o)));

        private Matrix Scalar(float v)
            => new(new float[,] { { v } },
                   new MatrixOptions { HasGrad = false, Device = device });

        public static (Matrix a, Matrix b) Broadcast(Matrix a, Matrix b)
        {
            int aw = a.shape.width;
            int ah = a.shape.height;

            int bw = b.shape.width;
            int bh = b.shape.height;

            bool awOk = aw == bw || aw == 1 || bw == 1;
            bool ahOk = ah == bh || ah == 1 || bh == 1;

            if (!awOk || !ahOk)
                throw new ArgumentException(
                    $"Incompatible broadcast shapes: A={a.shape}, B={b.shape}");

            Matrix A = a;
            Matrix B = b;

            if (aw == 1 && bw > 1)
                A = new Extend(FunctionType.Row).Apply((A, bw));

            if (bw == 1 && aw > 1)
                B = new Extend(FunctionType.Row).Apply((B, aw));

            if (ah == 1 && bh > 1)
                A = new Extend(FunctionType.Column).Apply((A, bh));

            if (bh == 1 && ah > 1)
                B = new Extend(FunctionType.Column).Apply((B, ah));

            return (A, B);
        }

        public Matrix GetRow(int i) => new GetVectors(FunctionType.Row).Apply((this, [i]));
        public Matrix GetColumn(int i) => new GetVectors(FunctionType.Column).Apply((this, [i]));
        public Matrix GetRows(int[] i) => new GetVectors(FunctionType.Row).Apply((this, i));
        public Matrix GetColumns(int[] i) => new GetVectors(FunctionType.Column).Apply((this, i));

        public Matrix RowConcatenate(Matrix o) => new Concatenate(FunctionType.Row, 2).Apply(new Matrix[] { this, o });
        public Matrix ColumnConcatenate(Matrix o) => new Concatenate(FunctionType.Column, 2).Apply(new Matrix[] { this, o });

        #endregion

        #region Operators

        public static Matrix operator >>(Matrix a, Matrix b) => a.MatMul(b);
        public static Matrix operator /(Matrix a, Matrix b) => a.Div(b);
        public static Matrix operator -(Matrix a, Matrix b) => a.Sub(b);
        public static Matrix operator +(Matrix a, Matrix b) => a.Add(b);
        public static Matrix operator *(Matrix a, Matrix b) => a.Mul(b);

        public static Matrix operator /(Matrix a, float b) => a.Div(b);
        public static Matrix operator -(Matrix a, float b) => a.Sub(b);
        public static Matrix operator +(Matrix a, float b) => a.Add(b);
        public static Matrix operator *(Matrix a, float b) => a.Mul(b);

        public static Matrix operator /(float a, Matrix b) => b.InvDiv(a);
        public static Matrix operator -(float a, Matrix b) => b.InvSub(a);
        public static Matrix operator +(float a, Matrix b) => b.Add(a);
        public static Matrix operator *(float a, Matrix b) => b.Mul(a);

        public static Matrix operator -(Matrix a) => a.Mul(-1);

        public static Matrix operator &(Matrix a, Matrix b) => a.RowConcatenate(b);
        public static Matrix operator |(Matrix a, Matrix b) => a.ColumnConcatenate(b);

        #endregion

        #region Indexers

        public Matrix this[int i] => GetRow(i);

        public Matrix this[int i, FunctionType type]
            => type == FunctionType.Row ? GetRow(i)
             : type == FunctionType.Column ? GetColumn(i)
             : throw new ArgumentException($"Function Type: {type} not allowed");

        public Matrix this[int i, char _] => GetRow(i);
        public Matrix this[char _, int i] => GetColumn(i);

        #endregion

        #region Generators

        public static ValuePackage Random(int w, int h)
        {
            var r = new Random();
            var m = new float[w, h];

            for (int x = 0; x < w; x++)
                for (int y = 0; y < h; y++)
                    m[x, y] = (float)((r.NextDouble() - 0.5) * 2);

            return new ValuePackage(m);
        }

        public static ValuePackage Identity(int size)
        {
            var m = new float[size, size];

            for (int x = 0; x < size; x++)
                for (int y = 0; y < size; y++)
                    m[x, y] = x == y ? 1 : 0;

            return new ValuePackage(m);
        }

        public static ValuePackage Zeros(int w, int h) => new(new float[w, h]);

        public static ValuePackage Ones(int w, int h)
        {
            var m = new float[w, h];
            for (int x = 0; x < w; x++)
                for (int y = 0; y < h; y++)
                    m[x, y] = 1;
            return new ValuePackage(m);
        }

        public static ValuePackage OneHot(int value, int categories, int width = 1)
        {
            var m = new float[width, categories];
            
            for (int x = 0; x < width; x++)
                for (int y = 0; y < categories; y++)
                    m[x, y] = y == value ? 1 : 0;

            return new ValuePackage(m);
        }

        #endregion
    }
}