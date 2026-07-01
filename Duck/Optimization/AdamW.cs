using Duck.Management;
using Duck.Matrix_Utilities;

namespace Duck.Optimization
{
    internal readonly struct ParameterData(int width, int height)
    {
        public readonly float[,] m = new float[width, height];
        public readonly float[,] v = new float[width, height];
        public readonly int width = width;
        public readonly int height = height;
    }
    public class AdamW : Optimizer
    {
        public readonly float lr;
        public readonly float b1;
        public readonly float b2;
        public readonly float e;
        public readonly float weightDecay;
        private int stepN = 0;
        public AdamW(Matrix[] parameters, float lr = 0.01f, float b1 = 0.9f, float b2 = 0.999f, float weightDecay = 0.01f, float e = 1e-8f) : base(parameters)
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
            float b1Correction = 1 - MathF.Pow(b1, stepN);
            float b2Correction = 1 - MathF.Pow(b2, stepN);
            float oneMinusB1 = 1 - b1;
            float oneMinusB2 = 1 - b2;
            float oneMinusWeightDecay = 1 - weightDecay;
            foreach (Matrix parameter in parameters)
            {
                ParameterData data = (ParameterData)parameter.matrixBase.optimizerData!;

                if (parameter.device == Device.CPU)
                {
                    MatrixCPU p = (MatrixCPU)parameter.matrixBase;

                    CPUManager.RunTask(0, parameter.shape.width, 0, parameter.shape.height, (x, y) =>
                    {
                        float gradient = p.GetGradient(x, y, parameter.transposed);

                        if (!float.IsRealNumber(gradient))
                            throw new Exception("Gradient is invalid");

                        data.m[x, y] = b1 * data.m[x, y] + oneMinusB1 * gradient;
                        data.v[x, y] = b2 * data.v[x, y] + oneMinusB2 * gradient * gradient;
                        float mHat = data.m[x, y] / b1Correction;
                        float vHat = data.v[x, y] / b2Correction;
                        p[x, y, parameter.transposed] *= oneMinusWeightDecay;
                        p[x, y, parameter.transposed] -= mHat / (MathF.Sqrt(vHat) + e) * lr;
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