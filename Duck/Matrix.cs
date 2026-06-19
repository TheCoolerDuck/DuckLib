using Duck.Functions.Basic;
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

        public Device device => matrixBase is MatrixCPU ? Device.CPU : Device.GPU;

        #endregion

        #region Initialization

        public Matrix(float[,] values, MatrixOptions options)
            : this(values, options, null) { }

        public Matrix(float[,] values)
            : this(values, new MatrixOptions(), null) { }

        internal Matrix(float[,] values, IBackwardContext? backwardContext)
            : this(values, new MatrixOptions(), backwardContext) { }

        internal Matrix(float[,] values, MatrixOptions options, IBackwardContext? backwardContext)
        {
            matrixBase =
                options.Device == Device.CPU
                    ? new MatrixCPU(values, options.HasGrad, backwardContext, options.Name)
                    : new MatrixGPU(values, backwardContext, options.Name);
        }

        internal Matrix((int width, int height) shape, CudaDeviceVariable<float> values, IBackwardContext ctx)
        {
            matrixBase = new MatrixGPU(shape, values, ctx);
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
                    : new MatrixGPU(values, null, name);
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

        public float[,] values => matrixBase.GetValues();
        public float[,] gradients => matrixBase.GetGradients();

        public void Backwards() => matrixBase.Backwards();
        public void ZeroGradient() => matrixBase.ZeroGradient();

        internal GPUMatrixStruct GPUValues()
            => new(this, ((MatrixGPU)matrixBase).values);

        internal GPUMatrixStruct GPUGradient()
            => new(this, ((MatrixGPU)matrixBase).gradient!);

        #endregion

        #region Formatting

        private const int DefaultSize = 8;

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
        public string InterfaceString(int resultSize)
        {
            StringBuilder sb = new();

            sb.AppendLine(
                $"Matrix {matrixBase.name}; ID: {matrixBase.ID:X}; Shape: {shape}"
            );

            sb.AppendLine("Values:");

            float[,] values = matrixBase.GetValues();

            AddMatrix((x, y) => transposed ? values[y, x] : values[x, y]);

            if (matrixBase.HasGradient())
            {
                sb.AppendLine("Gradient:");

                float[,] gradients = matrixBase.GetGradients();

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

        public string DataString(string format)
        {
            StringBuilder sb = new();

            sb.Append($"{matrixBase.name}:{shape.width},{shape.height}:{device}:{format}:");

            float[,] values = matrixBase.GetValues();

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
                if (f >= 10_000) return "#####"; 
                if (f >= 1_000) return ((int)f).ToString(); 
                if (f >= 100) return f.ToString("F1"); 
                if (f >= 10) return f.ToString("F2"); 
                
                return f.ToString("F3"); 
            } 
            else 
            { 
                if (f <= -1_000) return "-####"; 
                if (f <= -100) return ((int)f).ToString(); 
                if (f <= -10) return f.ToString("F1"); 
                
                return f.ToString("F2"); 
            } 
        }

        #endregion

        #region Matrix Ops (Function Objects)

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

        #region Operations

        public Matrix matMul(Matrix o) => matMulObj.Apply((this, o));
        public Matrix div(Matrix o) => divObj.Apply(Broadcast(this, o));
        public Matrix sub(Matrix o) => subObj.Apply(Broadcast(this, o));
        public Matrix add(Matrix o) => addObj.Apply(Broadcast(this, o));
        public Matrix mul(Matrix o) => mulObj.Apply(Broadcast(this, o));

        public Matrix div(float o) => divObj.Apply(Broadcast(this, Scalar(o)));
        public Matrix iDiv(float o) => divObj.Apply(Broadcast(Scalar(o), this));

        public Matrix sub(float o) => subObj.Apply(Broadcast(this, Scalar(o)));
        public Matrix iSub(float o) => subObj.Apply(Broadcast(Scalar(o), this));

        public Matrix add(float o) => addObj.Apply(Broadcast(this, Scalar(o)));
        public Matrix mul(float o) => mulObj.Apply(Broadcast(this, Scalar(o)));

        private Matrix Scalar(float v)
            => new(new float[,] { { v } },
                   new MatrixOptions { HasGrad = false, Device = device });

        public static (Matrix a, Matrix b) Broadcast(Matrix a, Matrix b)
        {
            bool isScalar = a.IsScalar() || b.IsScalar();
            bool isRowVector = (a.IsRowVector() || b.IsRowVector()) && b.shape.width == a.shape.width; 
            bool isColVector = (a.IsColVector() || b.IsColVector()) && b.shape.height == a.shape.height;
            bool isSame = a.shape == b.shape;
            if (!isScalar && !isRowVector && !isColVector && !isSame)
                throw new ArgumentException(
                    $"Matrices of incompatible shape: A: {a.shape}, B: {b.shape}");

            if (isSame)
                return (a, b);

            if (a.shape.width > b.shape.width && a.shape.height > b.shape.height)
                return (a, Extend.ApplyWhole(b, a.shape.width, a.shape.height));

            if (a.shape.width > b.shape.width)
                return (a, new Extend(FunctionType.Row).Apply((b, a.shape.width)));

            if (a.shape.height > b.shape.height)
                return (a, new Extend(FunctionType.Column).Apply((b, a.shape.height)));

            if (b.shape.width > a.shape.width && b.shape.height > a.shape.height)
                return (Extend.ApplyWhole(a, b.shape.width, b.shape.height), b);

            if (b.shape.width > a.shape.width)
                return (new Extend(FunctionType.Row).Apply((a, b.shape.width)), b);

            if (b.shape.height > a.shape.height)
                return (new Extend(FunctionType.Column).Apply((a, b.shape.height)), b);

            throw new Exception("im really not sure how we got here :/");
        }

        public Matrix getRow(int i) => getRowObj.Apply((this, [i]));
        public Matrix getCol(int i) => getColObj.Apply((this, [i]));
        public Matrix getRows(int[] i) => getRowObj.Apply((this, i));
        public Matrix getCols(int[] i) => getColObj.Apply((this, i));

        public Matrix rowConcatenate(Matrix o) => rowCatObj.Apply(new Matrix[] { this, o });
        public Matrix colConcatenate(Matrix o) => colCatObj.Apply(new Matrix[] { this, o });

        #endregion

        #region Operators

        public static Matrix operator <<(Matrix a, Matrix b) => a.matMul(b);
        public static Matrix operator /(Matrix a, Matrix b) => a.div(b);
        public static Matrix operator -(Matrix a, Matrix b) => a.sub(b);
        public static Matrix operator +(Matrix a, Matrix b) => a.add(b);
        public static Matrix operator *(Matrix a, Matrix b) => a.mul(b);

        public static Matrix operator /(Matrix a, float b) => a.div(b);
        public static Matrix operator -(Matrix a, float b) => a.sub(b);
        public static Matrix operator +(Matrix a, float b) => a.add(b);
        public static Matrix operator *(Matrix a, float b) => a.mul(b);

        public static Matrix operator /(float a, Matrix b) => b.iDiv(a);
        public static Matrix operator -(float a, Matrix b) => b.iSub(a);
        public static Matrix operator +(float a, Matrix b) => b.add(a);
        public static Matrix operator *(float a, Matrix b) => b.mul(a);

        public static Matrix operator -(Matrix a) => a.mul(-1);

        public static Matrix operator &(Matrix a, Matrix b) => a.rowConcatenate(b);
        public static Matrix operator |(Matrix a, Matrix b) => a.colConcatenate(b);

        #endregion

        #region Indexers

        public Matrix this[int i] => getRow(i);

        public Matrix this[int i, FunctionType type]
            => type == FunctionType.Row ? getRow(i)
             : type == FunctionType.Column ? getCol(i)
             : throw new ArgumentException($"Function Type: {type} not allowed");

        public Matrix this[int i, char _] => getRow(i);
        public Matrix this[char _, int i] => getCol(i);

        #endregion

        #region Generators

        public static float[,] Random(int w, int h, float denom = 1)
        {
            var r = new Random();
            var m = new float[w, h];

            for (int x = 0; x < w; x++)
                for (int y = 0; y < h; y++)
                    m[x, y] = (float)((r.NextDouble() - 0.5) * 2 / denom);

            return m;
        }

        public static float[,] Zeros(int w, int h) => new float[w, h];

        public static float[,] Ones(int w, int h)
        {
            var m = new float[w, h];
            for (int x = 0; x < w; x++)
                for (int y = 0; y < h; y++)
                    m[x, y] = 1;
            return m;
        }

        public static float[,] OneHot(int value, int categories, int width = 1)
        {
            var m = new float[width, categories];

            for (int y = 0; y < categories; y++)
                m[0, y] = y == value ? 1 : 0;

            return m;
        }

        #endregion
    }
}