
using Duck;
using Duck.Functional.Elementary;
using Duck.Functions.Basic;
using Duck.Functions.Value.Single;
using Duck.Modules;
using Duck.Modules.Activation;
using Duck.Modules.Basic;
using Duck.Modules.Normalization;
using Duck.Modules.PositionalEncoding;
using Duck.Tokenization;
using System.Threading.Tasks;
using System.Xml.Linq;

namespace ScratchLLM
{
    internal class SimpleLLMModel : IModule
    {
        private const int layers = 4;
        private const int vectorDims = 64;
        private const int key_queryDims = 16;
        private readonly int[] forwardMLP_size = [64, 128];
        private readonly int[] translationMLP_size = [64, 128];

        private readonly Matrix eigenVectors;
        private readonly Matrix[] keyMatrices;
        private readonly Matrix[] queryMatrices;
        private readonly Matrix[] valueMatrices;
        private readonly Sequential[] forwardMLPs;
        private readonly Sequential translationMLP;

        private readonly BLOS blos;

        private readonly SoftMax softMax = new();

        private readonly LinearTanh[] normilization;

        private readonly Matrix[] parameters;

        private readonly Mask maskNegInf = new(float.NegativeInfinity, MaskType.Tri);

        public SimpleLLMModel(Tokenizer tokenizer)
        {
            eigenVectors = new Matrix(Matrix.Random(tokenizer.tokenCount, vectorDims), new MatrixOptions() { Name = "eigen vectors"});

            blos = new(vectorDims);
            //rope = new RoPE(vectorDims, 1024);
            //LPE = new Matrix(Matrix.Random(1024, vectorDims));

            normilization = [.. Enumerable.Range(0, layers * 2 + 1).Select(i => new LinearTanh(vectorDims))];

            keyMatrices = [.. Enumerable.Range(0, layers).Select(i => new Matrix(Matrix.Random(vectorDims, key_queryDims) / MathF.Sqrt(vectorDims), new MatrixOptions() { Name = $"key-{i}" } ))]; 
            queryMatrices = [.. Enumerable.Range(0, layers).Select(i => new Matrix(Matrix.Random(vectorDims, key_queryDims) / MathF.Sqrt(vectorDims), new MatrixOptions() { Name = $"query-{i}" }))];
            valueMatrices = [.. Enumerable.Range(0, layers).Select(i => new Matrix(Matrix.Random(vectorDims, vectorDims) / MathF.Sqrt(vectorDims), new MatrixOptions() { Name = $"value-{i}" }))];

            {

                forwardMLPs = [.. Enumerable.Range(0, layers).Select(_ => {
                    List<IModule> modules = [];

                    int inSize = vectorDims;
                    int i = 0;
                    foreach (int h in forwardMLP_size)
                    {
                        modules.Add(new SwiGLU(inSize, h, "Forward " + _ + " SwiGLU " + i++));
                        inSize = h;
                    }

                    modules.Add(new Linear(inSize, vectorDims));

                    return new Sequential([.. modules]);
                })];
            }

            {
                List<IModule> modules = [];

                int inSize = vectorDims;
                int i = 0;
                foreach (int h in translationMLP_size)
                {
                    modules.Add(new SwiGLU(inSize, h, "Translation SwiGLU " + i++));
                    inSize = h;
                }

                modules.Add(new Linear(inSize, tokenizer.tokenCount));

                translationMLP = new Sequential([.. modules]);
            }

            parameters = [eigenVectors, ..normilization.SelectMany(m => m.GetParameters()), .. keyMatrices, .. queryMatrices, .. valueMatrices, .. forwardMLPs.SelectMany(m => m.GetParameters()), ..translationMLP.GetParameters()];
        }

        public Matrix TokensToVectors(int[] tokens)
        {
            return eigenVectors.GetRows(tokens);
        }

        public Matrix Forward(Matrix m)
        {
            m = normilization[^1].Forward(blos.Forward(m));

            for (int layer = 0; layer < layers; layer++)
            {
                {
                    Matrix n = normilization[layer * 2].Forward(m);

                    Matrix keys = n >> keyMatrices[layer];
                    Matrix queries = n >> queryMatrices[layer];
                    Matrix values = n >> valueMatrices[layer];

                    Matrix preweights = (queries >> keys.T()) / MathF.Sqrt(key_queryDims);

                    Matrix masked = maskNegInf.Apply(preweights);

                    Matrix weights = softMax.Forward(masked);

                    m += weights >> values;
                }

                {
                    Matrix n = normilization[layer * 2 + 1].Forward(m);

                    m += forwardMLPs[layer].Forward(n);
                }
            }

            return translationMLP.Forward(m);
        }

        public Matrix[] GetParameters()
        {
            return parameters;
        }
    }
}
