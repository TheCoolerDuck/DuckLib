using Duck.Functions.Value.Single;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.NVRTC;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Management
{
    internal static class GPUManager
    {
        private static CudaContext? context;
        public const int threads = 2048;
        public const int normalBlockSize = 256;
        public const int largeBlockSize = 1024;
        public static bool live => context != null;
        public static void Innit()
        {
            if (context == null)
            {
                context = new CudaContext(0);
            }
        }

        public static void SetKernelSize(CudaKernel kernel, int threads)
        {
            kernel.BlockDimensions = normalBlockSize;
            kernel.GridDimensions = (threads + normalBlockSize - 1) / normalBlockSize;
        }

        public static CudaKernel Compile([CallerFilePath] string file = "")
        {
            if (!live)
                Innit();

            string source = File.ReadAllText(file[..^3] + ".cu");

            return _Compile(source);
        }

        public static CudaKernel CompileGradient([CallerFilePath] string file = "")
        {
            if (!live)
                Innit();

            string source = File.ReadAllText(file[..^3] + "Gradient.cu");

            return _Compile(source);
        }

        public static CudaKernel Compile(string parameters, string returnType, string defaultCode, Type functionType, [CallerFilePath] string file = "")
        {
            if (!live)
                Innit();

            string source = FunctionImport(parameters, returnType, defaultCode, functionType, false) + File.ReadAllText(file[..^3] + ".cu");

            return _Compile(source);
        }

        public static CudaKernel CompileGradient(string parameters, string returnType, string defaultCode, Type functionType, [CallerFilePath] string file = "")
        {
            if (!live)
                Innit();

            string source = FunctionImport(parameters, returnType, defaultCode, functionType, true) + File.ReadAllText(file[..^3] + "Gradient.cu");

            return _Compile(source);
        }

        private static CudaKernel _Compile(string source)
        {
            CudaRuntimeCompiler rtc = new(source, "Main");
            try
            {
                rtc.Compile(["--gpu-architecture=compute_86"]);
            }
            catch (NVRTCException)
            {
                throw new Exception($"Kernel failed to compile:\n{rtc.GetLogAsString()}");
            }
            byte[] ptx = rtc.GetPTX();
            rtc.Dispose();
            return context!.LoadKernelPTX(ptx, "Main");
        }

        private static string FunctionImport(string parameters, string returnType, string defaultCode, Type functionType, bool darivative)
        {
            StringBuilder sb = new();

            sb.AppendLine($@"
__device__ {returnType} apply({parameters}, int ID)
{{
    switch(ID)
    {{");
            List<Type> types = [.. GetAllTypesThatImplementInterface(functionType)];

            foreach (Type t in types)
            {
                string code = (string)t
                    .GetMethod(darivative ? "GetGPUApplyDerivative" : "GetGPUApply")!
                    .Invoke(null, null)!;

                sb.AppendLine($"        case {StableHash(t.Name)}: {code}");
            }

            sb.Append($@"        default: return {defaultCode};
    }}
}}

");
            return sb.ToString();
        }

        public static IEnumerable<Type> GetAllTypesThatImplementInterface(Type iface)
        {
            return AppDomain.CurrentDomain
                .GetAssemblies()
                .SelectMany(a => a.GetTypes())
                .Where(t =>
                    !t.IsInterface &&
                    !t.IsAbstract &&
                    iface.IsAssignableFrom(t));
        }

        public static int StableHash(string s)
        {
            unchecked
            {
                const int offset = unchecked((int)2166136261);
                const int prime = 16777619;

                int hash = offset;

                foreach (char c in s)
                {
                    hash ^= c;
                    hash *= prime;
                }

                return hash;
            }
        }

        public static void Dispose()
        {
            context?.Dispose();
            context = null;
        }
    }
}
