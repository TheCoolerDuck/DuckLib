using Duck.Functions.Basic;
using Duck.Functions.Parameters;
using Duck.Management;
using Duck.Matrix_Utilities;
using Duck.Modules.Advanced;

namespace Duck.Modules.Loss
{
    public class CrossEntropy : ILoss<MatrixAndIndexArray>
    {
        private readonly static SoftMax softMax = new(FunctionType.Column);
        public Matrix Apply(MatrixAndIndexArray p)
        {
            if (p.m.device == Device.CPU)
            {
                Matrix probs = softMax.Forward(p.m);

                float[,] n = new float[probs.shape.width, 1];

                MatrixCPU m = (MatrixCPU)p.m.matrixBase;
                MatrixCPU pro = (MatrixCPU)probs.matrixBase;

                CPUManager.RunTask(0, probs.shape.width, i =>
                {
                    n[i, 0] = -MathF.Log(pro[i, p.i[i], probs.transposed] + 1e-8f);
                });

                CPUManager.RunTask(0, p.m.shape.width, 0, p.m.shape.height, (i, j) =>
                {
                    bool isHot = j == p.i[i];
                    m.AddGradient(i, j, isHot ? pro[i, j, probs.transposed] - 1 : pro[i, j, probs.transposed], p.m.transposed);
                });

                m.usages.Clear();

                p.result = new(n, new BackwardContext<MatrixAndIndexArray>(null, p));
            }
            else
            {
                throw new NotImplementedException();
            }

            return p.result;
        }
    }
}