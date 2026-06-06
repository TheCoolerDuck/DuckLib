using Duck.CustomLLM.Library.Objects.MatrixObjects;
using Duck.Functions.Parameters;
using Duck.Functions.Value.Single;
using Duck.Management;
using Duck.Matrix_Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Functions.Basic
{
    internal class ElementFunction<T> :IBasicFunction<SingleMatrix> where T : ISingleValueFunction
    {
        public Matrix Apply(SingleMatrix p)
        {

            if (p.m.device == Device_Management.Device.CPU)
            {
                float[,] values = new float[p.m.shape.width, p.m.shape.height];

                CPUManager.RunTask(0, p.m.shape.width, 0, p.m.shape.height, (x, y) =>
                {
                    values[x, y] = T.Apply(((MatrixCPU)p.m.matrixBase)[x, y, p.m.transposed]);
                });

                p.result = new Matrix(values, new BackwardContext<SingleMatrix>(this, p), Device_Management.Device.CPU);
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
                float[,] values = new float[p.m.shape.width, p.m.shape.height];

                MatrixCPU m = (MatrixCPU)p.m.matrixBase;
                MatrixCPU result = (MatrixCPU)p.result.matrixBase;

                CPUManager.RunTask(0, p.m.shape.width, 0, p.m.shape.height, (x, y) =>
                {
                    float rGrad = result.GetGradient(x, y, p.result.transposed);
                    float mGrad = T.ApplyDerivative(m[x, y, p.m.transposed]);
                    m.AddGradient(x, y, mGrad * rGrad, p.m.transposed);
                });
            }
            else
            {
                throw new NotImplementedException();
            }
        }
    }
}
