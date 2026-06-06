using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.NVRTC;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Management
{
    internal static class GPUManager
    {
        private static CudaContext? _context;
        public const int threads = 2048;
        public static CudaContext Context
        {
            get
            {
                _context ??= new CudaContext(0);
                return _context;
            }
        }

        public static void Ready()
        {
            _context ??= new CudaContext(0);
        }

        public static CudaKernel Compile(string source)
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
            return Context.LoadKernelPTX(ptx, "Main");
        }

        public static IEnumerable<Type> GetAllTypesThatImplementInterface<T>()
        {
            return Assembly.GetExecutingAssembly()
                .GetTypes()
                .Where(type => typeof(T).IsAssignableFrom(type) && !type.IsInterface && !type.IsAbstract);
        }

        public static void Dispose()
        {
            _context?.Dispose();
            _context = null;
        }
    }
}
