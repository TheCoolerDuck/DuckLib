using Duck.Functions.Parameters;
using Duck.Functions.Value.Single;
using Duck.Management;
using Duck.Matrix_Utilities;
using ManagedCuda;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Basic
{
    internal class ElementFunction<T> : IBasicFunction<SingleMatrix> where T : ISingleValueFunction
    {
        protected override Matrix ApplyCPU(SingleMatrix p)
        {
            float[,] values = new float[p.m.shape.width, p.m.shape.height];

            CPUManager.RunTask(0, p.m.shape.width, 0, p.m.shape.height, (x, y) =>
            {
                values[x, y] = T.Apply(((MatrixCPU)p.m.matrixBase)[x, y, p.m.transposed]);
            });

            p.result = new Matrix(values, new MatrixOptions() { Device = Device.CPU }, new BackwardContext<SingleMatrix>(this, p));

            return p.result;
        }

        protected override void ApplyGradientCPU(SingleMatrix p)
        {
            MatrixCPU m = (MatrixCPU)p.m.matrixBase;
            MatrixCPU result = (MatrixCPU)p.result!.matrixBase;

            CPUManager.RunTask(0, p.m.shape.width, 0, p.m.shape.height, (x, y) =>
            {
                float rGrad = result.GetGradient(x, y, p.result.transposed);
                float mGrad = T.ApplyDerivative(m[x, y, p.m.transposed]);
                m.AddGradient(x, y, mGrad * rGrad, p.m.transposed);
            });
        }
        protected override Matrix ApplyGPU(SingleMatrix p)
        {
            int size = p.m.shape.width * p.m.shape.height;
            CudaDeviceVariable<float> values = new(size);
            p.result = new Matrix((p.m.shape.width, p.m.shape.height), values, new BackwardContext<SingleMatrix>(this, p));

            GPUMatrixStruct a = p.m.GPUValues();
            GPUMatrixStruct b = p.result.GPUValues();

            applyKernel.BlockDimensions = 64;
            applyKernel.GridDimensions = (64 + size - 1) / 64;

            applyKernel.Run(a, b, GPUManager.StableHash(typeof(T).Name));

            return p.result;
        }

        protected override void ApplyGradientGPU(SingleMatrix p)
        {
            GPUMatrixStruct a = p.m.GPUValues();
            GPUMatrixStruct ag = p.m.GPUGradient();
            GPUMatrixStruct bg = p.result!.GPUGradient();

            int size = a.width * a.height;

            gradientKernel.BlockDimensions = 64;
            gradientKernel.GridDimensions = (64 + size - 1) / 64;

            gradientKernel.Run(a, ag, bg, GPUManager.StableHash(typeof(T).Name));
        }
        static ElementFunction()
        {
            MakeFunctionImport();
        }

        private static CudaKernel? _applyKernel;
        public static CudaKernel applyKernel => _applyKernel ??= GPUManager.Compile(File.ReadAllText("Functions\\GPUCode\\ElementFunction.cu"));

        private static CudaKernel? _gradientKernel;
        public static CudaKernel gradientKernel => _gradientKernel ??= GPUManager.Compile(File.ReadAllText("Functions\\GPUCode\\ElementFunctionGradient.cu"));
        private static void MakeFunctionImport()
        {
            StringBuilder sb = new();

            sb.AppendLine(@"
__device__ float apply(float x, int ID)
{
    switch(ID)
    {");
            List<Type> types = [.. GPUManager.GetAllTypesThatImplementInterface(typeof(ISingleValueFunction))];

            foreach (Type t in  types)
            {
                string code = (string)t
                    .GetMethod("GetGPUApply")!
                    .Invoke(null, null)!;

                string darivative = (string)t
                    .GetMethod("GetGPUApplyDerivative")!
                    .Invoke(null, null)!;

                sb.AppendLine($"        case {GPUManager.StableHash(t.Name)}: return {code};");
                sb.AppendLine($"        case {-GPUManager.StableHash(t.Name)}: return {darivative};");
            }

            sb.Append(@"        default: return 1.0f / 0.0f;
    }
}
");
            File.WriteAllText("C:\\Users\\pjsol\\source\\repos\\DuckLib\\Duck\\Functions\\GPUCode\\ElementFuntionsApply.h", sb.ToString());
        }

        protected override void ValidateParameters(SingleMatrix p) { }
    }
}
