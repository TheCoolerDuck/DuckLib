using Duck.Management;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Parameters
{
    public interface IParameter
    {
        public Device GetOperationDevice()
        {
            Matrix[] matrices = MatricesUsed();

            if (matrices.Length == 0)
                return Device.Unspecified;

            Device device = matrices[0].device;

            foreach (Matrix matrix in matrices)
                if (matrix.device != device)
                    throw new Exception("Matrices must have the same device");

            return device;
        }
        public Matrix? result { get; set; }
        public Matrix[] MatricesUsed();
    }
}
