using Duck.CustomLLM.Library.Objects.MatrixObjects;
using Duck.Management;
using Duck.Matrix_Utilities;

namespace Duck.Optimization
{
    internal readonly struct ParameterData(int width, int height)
    {
        public readonly double[,] m = new double[width, height];
        public readonly double[,] v = new double[width, height];
        public readonly int width = width;
        public readonly int height = height;
    }
    public class AdamW : Optimizer
    {
        public readonly double lr;
        public readonly double b1;
        public readonly double b2;
        public readonly double e;
        public readonly double weightDecay;
        private int stepN = 0;
        public AdamW(Matrix[] parameters, double lr = 0.01, double b1 = 0.9, double b2 = 0.999, double weightDecay = 0.01, double e = 1e-8) : base(parameters)
        {
            this.lr = lr;
            this.b1 = b1;
            this.b2 = b2;
            this.e = e;
            this.weightDecay = weightDecay;
            foreach (Matrix parameter in parameters)
                parameter.matrixBase.optimizerData = new ParameterData(parameter.shape.width, parameter.shape.height);
        }
        public override void step()
        {
            stepN++;
            double b1Correction = 1 - Math.Pow(b1, stepN);
            double b2Correction = 1 - Math.Pow(b2, stepN);
            double oneMinusB1 = 1 - b1;
            double oneMinusB2 = 1 - b2;
            double oneMinusWeightDecay = 1 - weightDecay;
            foreach (Matrix parameter in parameters)
            {
                ParameterData data = (ParameterData)parameter.matrixBase.optimizerData!;

                if (parameter.device == Device_Management.Device.CPU)
                {
                    MatrixCPU p = (MatrixCPU)parameter.matrixBase;

                    CPUThreadManager.RunTask(0, parameter.shape.width, 0, parameter.shape.height, (x, y) =>
                    {
                        double gradient = p.GetGradient(x, y, parameter.transposed);
                        data.m[x, y] = b1 * data.m[x, y] + oneMinusB1 * gradient;
                        data.v[x, y] = b2 * data.v[x, y] + oneMinusB2 * gradient * gradient;
                        double mHat = data.m[x, y] / b1Correction;
                        double vHat = data.v[x, y] / b2Correction;
                        p[x, y, parameter.transposed] = (p[x, y, parameter.transposed] - mHat / (Math.Sqrt(vHat) + e) * lr) * oneMinusWeightDecay;
                    });
                }
                else
                {
                    throw new NotImplementedException();
                }
            }
        }
    }
}