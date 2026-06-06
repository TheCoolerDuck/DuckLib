using Duck.CustomLLM.Library.Objects.MatrixObjects;
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
        private readonly MaskType maskType;
        public Matrix Apply(SingleMatrix p)
        {
            if (p.m.device == Device_Management.Device.CPU)
            {
                if (maskType == MaskType.Tri)
                {
                    if (p.m.shape.width != p.m.shape.height)
                        throw new ArgumentException("Width must equal height for tri mask");

                    float[,] values = new float[p.m.shape.width, p.m.shape.height];

                    MatrixCPU m = (MatrixCPU)p.m.matrixBase;

                    CPUManager.RunTask(0, p.m.shape.width, 0, p.m.shape.height, (x, y) =>
                    {
                        if (y > x)
                            values[x, y] = maskValue;
                        else
                            values[x, y] = m[x, y, p.m.transposed];
                    });

                    p.result = new Matrix(values, new BackwardContext<SingleMatrix>(this, p), Device_Management.Device.CPU);
                }
                else
                {
                    throw new NotImplementedException();
                }
            }
            else
            {
                throw new NotImplementedException();
            }

            return p.result;
        }

        public void ApplyGradient(SingleMatrix p)
        {
            if (p.result == null)
                throw new ArgumentException("Params must have a result");

            if (p.m.device == Device_Management.Device.CPU)
            {
                if (maskType == MaskType.Tri)
                {
                    if (p.m.shape.width != p.m.shape.height)
                        throw new ArgumentException("Width must equal height for tri mask");

                    MatrixCPU m = (MatrixCPU)p.m.matrixBase;
                    MatrixCPU r = (MatrixCPU)p.result.matrixBase;

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
            else
            {
                throw new NotImplementedException();
            }
        }
    }
}
