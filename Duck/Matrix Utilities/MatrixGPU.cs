using Duck.Management;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Matrix_Utilities
{
    internal struct GPUMatrixStruct(Matrix matrix, CudaDeviceVariable<float> values)
	{
        public CUdeviceptr values = values.DevicePointer;
        public int width = matrix.shape.width;
        public int height = matrix.shape.height;
        public bool transposed = matrix.transposed;

	}

    internal class MatrixGPU : MatrixBase
    {
        public readonly CudaDeviceVariable<float> values;
        public readonly CudaDeviceVariable<float>? gradient;
        public new readonly (int width, int height) shape;
        internal MatrixGPU(float[,] values, IBackwardContext? backwardContext = null, string name = "") : base(backwardContext, name)
        {
            GPUManager.Ready();
            shape = (values.GetLength(0), values.GetLength(1));
            this.values = new CudaDeviceVariable<float>(shape.width * shape.height);

            float[] flat = new float[shape.width * shape.height];
            for (int y = 0; y < shape.height; y++)
                for (int x = 0; x < shape.width; x++)
                    flat[y * shape.width + x] = values[x, y];


            this.values.CopyToDevice(flat);
            gradient = null;
        }
        internal MatrixGPU((int width, int height) shape, CudaDeviceVariable<float> values, IBackwardContext backwardContext) : base(backwardContext, "")
        {
            GPUManager.Ready();
            this.shape = shape;
            this.values = values;
            gradient = new CudaDeviceVariable<float>(shape.width * shape.height);
        }


        public override (int width, int height) GetShape()
        {
            return shape;
        }

        public override bool HasGradient()
        {
            return gradient != null;
        }

        protected override void ZeroGradientValues()
        {
            throw new NotImplementedException();
        }

        internal override float[,] GetValues()
        {
            float[] flat = new float[shape.width * shape.height];
            float[,] output = new float[shape.width, shape.height];

            values.CopyToHost(flat);

            for (int y = 0; y < shape.height; y++)
            {
                for (int x = 0; x < shape.width; x++)
                {
                    output[x, y] = flat[y * shape.width + x];
                }
            }

            return output;
        }

        internal override float[,] GetGradients()
        {
            if (gradient == null)
                throw new ArgumentNullException();

            float[,] output = new float[shape.width, shape.height];
            gradient.CopyToHost(output);
            return output;
        }
    }
}
