using Duck.CustomLLM.Library.Objects.MatrixObjects;
using Duck.Functions.Parameters;
using Duck.Management;
using Duck.Matrix_Utilities;

namespace Duck.Functions.Basic
{
    public class Concatenate(FunctionType type) : IBasicFunction<MatrixArray>
    {
        public FunctionType type = Validate(type);

        private static FunctionType Validate(FunctionType t)
        {
            if (t == FunctionType.Whole)
                throw new ArgumentException("FunctionType.Whole is not allowed.", nameof(t));
            return t;
        }

        public Matrix Apply(MatrixArray p)
        {
            if (p.a == null || p.a.Length == 0)
                throw new ArgumentException("Matrix array must not be empty.");

            // Validate shapes and devices
            for (int i = 1; i < p.a.Length; i++)
            {
                if (type == FunctionType.Row && p.a[i].shape.width != p.a[0].shape.width)
                    throw new ArgumentException($"Matrix shapes incompatible for row concatenation at index {i}: {p.a[i].shape} | {p.a[0].shape}");
                if (type == FunctionType.Column && p.a[i].shape.height != p.a[0].shape.height)
                    throw new ArgumentException($"Matrix shapes incompatible for column concatenation at index {i}: {p.a[i].shape} | {p.a[0].shape}");
                if (p.a[i].device != p.a[0].device)
                    throw new ArgumentException($"All matrices must be on the same device (mismatch at index {i})");
            }

            if (p.a[0].device != Device_Management.Device.CPU)
                throw new NotImplementedException();

            // Compute output shape — sum along the concat axis, fixed on the other
            int nw = type == FunctionType.Column
                ? p.a.Sum(m => m.shape.width)
                : p.a[0].shape.width;
            int nh = type == FunctionType.Row
                ? p.a.Sum(m => m.shape.height)
                : p.a[0].shape.height;

            // Precompute each matrix's offset along the concat axis
            int[] offsets = new int[p.a.Length];
            for (int i = 1; i < p.a.Length; i++)
                offsets[i] = offsets[i - 1] + (type == FunctionType.Column
                    ? p.a[i - 1].shape.width
                    : p.a[i - 1].shape.height);

            MatrixCPU[] srcs = p.a.Select(m => (MatrixCPU)m.matrixBase).ToArray();
            float[,] values = new float[nw, nh];

            CPUManager.RunTask(0, nw, 0, nh, (x, y) =>
            {
                int idx = GetSourceIndex(x, y, offsets);
                MatrixCPU src = srcs[idx];
                int sx = type == FunctionType.Column ? x - offsets[idx] : x;
                int sy = type == FunctionType.Row ? y - offsets[idx] : y;
                values[x, y] = src[sx, sy, p.a[idx].transposed];
            });

            p.result = new Matrix(values, new BackwardContext<MatrixArray>(this, p));
            return p.result;
        }

        public void ApplyGradient(MatrixArray p)
        {
            if (p.result == null)
                throw new ArgumentException("Params must have a result");
            if (p.a[0].device != Device_Management.Device.CPU)
                throw new NotImplementedException();

            // Precompute offsets (same logic as Apply)
            int[] offsets = new int[p.a.Length];
            for (int i = 1; i < p.a.Length; i++)
                offsets[i] = offsets[i - 1] + (type == FunctionType.Column
                    ? p.a[i - 1].shape.width
                    : p.a[i - 1].shape.height);

            MatrixCPU[] dsts = p.a.Select(m => (MatrixCPU)m.matrixBase).ToArray();
            MatrixCPU r = (MatrixCPU)p.result.matrixBase;

            CPUManager.RunTask(0, p.result.shape.width, 0, p.result.shape.height, (x, y) =>
            {
                int idx = GetSourceIndex(x, y, offsets);
                MatrixCPU dst = dsts[idx];
                int sx = type == FunctionType.Column ? x - offsets[idx] : x;
                int sy = type == FunctionType.Row ? y - offsets[idx] : y;
                dst.AddGradient(sx, sy, r.GetGradient(x, y, p.result.transposed), p.a[idx].transposed);
            });
        }

        /// <summary>
        /// Given output coordinates (x, y), returns which source matrix index they belong to.
        /// </summary>
        private int GetSourceIndex(int x, int y, int[] offsets)
        {
            int coord = type == FunctionType.Column ? x : y;
            int idx = 0;
            // Walk forward while the next matrix's offset is still <= coord
            while (idx + 1 < offsets.Length && offsets[idx + 1] <= coord)
                idx++;
            return idx;
        }
    }
}