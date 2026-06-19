using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Management
{
    public enum Device
    {
        CPU,
        GPU,
        Unspecified 
    }
    public static class DeviceManager
    {
        public static Device defaultDevice = Device.CPU;
    }
}
