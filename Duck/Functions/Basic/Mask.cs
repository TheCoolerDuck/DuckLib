using Duck.Functions.Parameters;
using Duck.Management;
using Duck.Matrix_Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Basic
{
    public enum MaskType
    {
        Tri
    }
    public class Mask(float maskValue, MaskType maskType) : IBasicFunction<SingleMatrix>
    {
        private readonly float maskValue = maskValue;
        private readonly MaskType maskType = maskType;

        protected override Matrix ApplyCPU(SingleMatrix p)
        {
            if (maskType == MaskType.Tri)
            {
                float[,] values = new float[p.m.shape.width, p.m.shape.height];

                MatrixCPU m = (MatrixCPU)p.m.matrixBase;

                CPUManager.RunTask(0, p.m.shape.width, 0, p.m.shape.height, (x, y) =>
                {
                    if (y > x)
                        values[x, y] = maskValue;
                    else
                        values[x, y] = m[x, y, p.m.transposed];
                });

                p.result = new Matrix(values, new MatrixOptions() { Device = Device.CPU }, new BackwardContext<SingleMatrix>(this, p));
            }
            else
            {
                throw new NotImplementedException();
            }

            return p.result;
        }

        protected override Matrix ApplyGPU(SingleMatrix p)
        {
            throw new NotImplementedException();
        }

        protected override void ApplyGradientCPU(SingleMatrix p)
        {
            if (maskType == MaskType.Tri)
            {
                MatrixCPU m = (MatrixCPU)p.m.matrixBase;
                MatrixCPU r = (MatrixCPU)p.result!.matrixBase;

                CPUManager.RunTask(0, p.m.shape.width, 0, p.m.shape.height, (x, y) =>
                {
                    if (x >= y)
                        m.AddGradient(x, y, r.GetGradient(x, y, p.result.transposed), p.m.transposed);
                });
            }
            else
            {
                throw new NotImplementedException();
            }
        }

        protected override void ApplyGradientGPU(SingleMatrix p)
        {
            throw new NotImplementedException();
        }

        protected override void ValidateParameters(SingleMatrix p)
        {
            if (maskType == MaskType.Tri)
            {
                if (p.m.shape.width != p.m.shape.height)
                    throw new ArgumentException("Width must equal height for tri mask");
            }
            else
            {
                throw new NotImplementedException();
            }
        }
    }
}
