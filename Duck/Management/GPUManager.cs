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
            _context?.Dispose();
            _context = null;
        }
    }
}
