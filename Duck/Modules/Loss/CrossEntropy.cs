using Duck.CustomLLM.Library.Objects.MatrixObjects;
using Duck.Functions.Basic;
using Duck.Functions.Parameters;
using Duck.Management;
using Duck.Matrix_Utilities;
using Duck.Modules.Advanced;

namespace Duck.Modules.Loss
{
    public class CrossEntropy : ILoss<DoubleMatrix>
    {
        private readonly static SoftMax softMax = new(FunctionType.Column);
        public Matrix Apply(DoubleMatrix p)
        {
            if (p.a.device != p.b.device)
                throw new ArgumentException("Matrices must be on the same device");

            if (p.a.device == Device_Management.Device.CPU)
            {

                Matrix probs = softMax.Forward(p.a);

                double[,] n = new double[probs.shape.width, 1];

                MatrixCPU m = (MatrixCPU)p.a.matrixBase;
                MatrixCPU pro = (MatrixCPU)probs.matrixBase;
                MatrixCPU ind = (MatrixCPU)p.b.matrixBase;

                CPUThreadManager.RunTask(0, probs.shape.width, i =>
                {
                    int j = (int)Math.Round(ind[0, i, p.b.transposed]);
                    n[i, 0] = -Math.Log(pro[i, j, probs.transposed] + 1e-8f);
                });

                CPUThreadManager.RunTask(0, p.a.shape.width, 0, p.a.shape.height, (i, j) =>
                {
                    bool isHot = j == (int)Math.Round(ind[0, i, p.b.transposed]);
                    m.AddGradient(i, j, isHot ? pro[i, j, probs.transposed] - 1 : pro[i, j, probs.transposed], p.a.transposed);
                });

                m.usages.Clear();

                p.result = new(n, new BackwardContext<DoubleMatrix>(null, p));
            }
            else
            {
                throw new NotImplementedException();
            }

            return p.result;
        }
    }
}