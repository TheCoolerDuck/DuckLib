using Duck.Functional.Parameters;
using Duck.Functions.Parameters;
using Duck.Functions.Value.Double;
using Duck.Functions.Value.Single;
using Duck.Management;
using Duck.Matrix_Utilities;
using ManagedCuda;
using ManagedCuda.VectorTypes;
using System;
using System.Drawing;
using System.Text;

namespace Duck.Functional.Elementary
{
    public class MatrixFunction<T> : IBasicFunction<DoubleMatrix> where T : IDoubleValueFunction
    {
        protected override Matrix ApplyCPU(DoubleMatrix p)
        {
            MatrixCPU aCPU = (MatrixCPU)p.a.matrixBase;
            MatrixCPU bCPU = (MatrixCPU)p.b.matrixBase;

            float[,] values = new float[p.a.shape.width, p.a.shape.height];

            CPUManager.RunTask(0, p.a.shape.width, 0, p.a.shape.height, (x, y) =>
            {
                values[x, y] = T.Apply(aCPU[x, y, p.a.transposed], bCPU[x, y, p.b.transposed]);
            });

            p.result = new Matrix(values, new MatrixOptions() { Device = Device.CPU, HasGrad = ((IParameter)p).ResultHasGradient() }, new BackwardContext<DoubleMatrix>(this, p));

            return p.result;
        }

        protected override void ApplyGradientCPU(DoubleMatrix p)
        {
            MatrixCPU aCPU = (MatrixCPU)p.a.matrixBase;
            MatrixCPU bCPU = (MatrixCPU)p.b.matrixBase;
            MatrixCPU result = (MatrixCPU)p.result!.matrixBase;

            CPUManager.RunTask(0, p.result.shape.width, 0, p.result.shape.height, (x, y) =>
            {
                float aVal = aCPU[x, y, p.a.transposed];
                float bVal = bCPU[x, y, p.b.transposed];
                float rGrad = result.GetGradient(x, y, p.result.transposed);

                (float aGrad, float bGrad) = T.ApplyDerivative(aVal, bVal);

                aCPU.AddGradient(x, y, aGrad * rGrad, p.a.transposed);
                bCPU.AddGradient(x, y, bGrad * rGrad, p.b.transposed);
            });
        }

        protected override Matrix ApplyGPU(DoubleMatrix p)
        {
            CudaDeviceVariable<float> values = new(p.a.shape.width * p.a.shape.height);
            p.result = new Matrix((p.a.shape.width, p.a.shape.height), values, new BackwardContext<DoubleMatrix>(this, p), ((IParameter)p).ResultHasGradient());

            applyKernel.BlockDimensions = 64;
            applyKernel.GridDimensions = (p.a.size + 63) / 64;

            applyKernel.Run(p.a.GPUValues(), p.b.GPUValues(), p.result.GPUValues(), GPUManager.StableHash(typeof(T).Name));

            return p.result;
        }

        protected override void ApplyGradientGPU(DoubleMatrix p)
        {
            gradientKernel.BlockDimensions = 64;
            gradientKernel.GridDimensions = (p.a.size + 63) / 64;

            gradientKernel.Run(p.a.GPUValues(), p.b.GPUValues(), p.a.GPUGradient(), p.b.GPUGradient(), p.result!.GPUGradient(), GPUManager.StableHash(typeof(T).Name));
        }

        static MatrixFunction()
        {
            MakeFunctionImports();
        }

        private static CudaKernel? _applyKernel;
        public static CudaKernel applyKernel => _applyKernel ??= GPUManager.Compile(File.ReadAllText("Functional\\GPUCode\\MatrixFunction.cu"));

        private static CudaKernel? _gradientKernel;
        public static CudaKernel gradientKernel => _gradientKernel ??= GPUManager.Compile(File.ReadAllText("Functional\\GPUCode\\ElementFunctionGradient.cu"));
        private static void MakeFunctionImports()
        {
            List<Type> types = [.. GPUManager.GetAllTypesThatImplementInterface(typeof(IDoubleValueFunction))];

            StringBuilder forward = new();
            StringBuilder backward = new();



            forward.AppendLine(@"
__device__ float apply(float x, float y, int ID)
{
    switch(ID)
    {");

            backward.AppendLine(@"
__device__ float2 apply(float x, float y, int ID)
{
    switch(ID)
    {");


            foreach (Type t in types)
            {
                string codeF = (string)t
                    .GetMethod("GetGPUApply")!
                    .Invoke(null, null)!;

                string codeB = (string)t
                    .GetMethod("GetGPUApplyDerivative")!
                    .Invoke(null, null)!;

                forward.AppendLine($"        case {GPUManager.StableHash(t.Name)}: return {codeF};");
                backward.AppendLine($"        case {GPUManager.StableHash(t.Name)}: return {codeB};");
            }

            forward.Append(@"        default: return 1.0f / 0.0f;
    }
}
");
            backward.Append(@"        default: return make_float2(1.0f / 0.0f, 1.0f / 0.0f);
    }
}
");

            File.WriteAllText("C:\\Users\\pjsol\\source\\repos\\DuckLib\\Duck\\Functional\\GPUCode\\MatrixFunctionsApply.h", forward.ToString());
            File.WriteAllText("C:\\Users\\pjsol\\source\\repos\\DuckLib\\Duck\\Functional\\GPUCode\\MatrixFunctionsApplyGradient.h", backward.ToString());

        }

        protected override void ValidateParameters(DoubleMatrix p)
        {
            if (p.a.shape != p.b.shape)
                throw new ArgumentException($"Matrices must be of same shape {p.a} {p.b}");
        }
    }
}