
using Duck.CustomLLM.Library.Objects.MatrixObjects;
using Duck.Functions.Basic;
using Duck.Functions.Value.Single;
using Duck.Modules;
using Duck.Modules.Activation;
using Duck.Modules.Advanced;
using Duck.Modules.Basic;
using Duck.Tokenization;
using System.Threading.Tasks;

namespace ScratchLLM
{
    internal class SimpleLLMModel : IModule
    {
        private const int layers = 8;
        private const int vectorDims = 16;
        private const int key_queryDims = 4;

        private readonly int[] forwardMLP_size = [16, 32];
        private readonly int[] translationMLP_size = [16, 32];

        private readonly IModule activation = new Swish();

        private readonly Matrix eigenVectors;
        private readonly Matrix[] keyMatrices;
        private readonly Matrix[] queryMatrices;
        private readonly Matrix[] valueMatrices;
        private readonly Sequential[] forwardMLPs;
        private readonly Sequential translationMLP;

        private readonly SoftMax softMax = new(FunctionType.Column);

        private readonly Apply<TanH> normilization = new();

        private readonly BLOS positionalEncoding = new(vectorDims);

        private readonly Matrix[] parameters;

        private readonly Mask maskNegInf = new(double.NegativeInfinity, MaskType.Tri);

        private readonly Concatenate concatenate = new(FunctionType.Row);

        public SimpleLLMModel(Tokenizer tokenizer)
        {
            eigenVectors = new Matrix(Matrix.Random(vectorDims, tokenizer.tokenCount));

            keyMatrices = [.. Enumerable.Range(0, layers).Select(i => new Matrix(Matrix.Random(vectorDims, key_queryDims, vectorDims), $"key-{i}"))]; 
            queryMatrices = [.. Enumerable.Range(0, layers).Select(i => new Matrix(Matrix.Random(vectorDims, key_queryDims, vectorDims), $"query-{i}"))];
            valueMatrices = [.. Enumerable.Range(0, layers).Select(i => new Matrix(Matrix.Random(vectorDims, vectorDims, vectorDims), $"value-{i}"))];

            forwardMLPs = [..Enumerable.Range(0, layers).Select(i => Sequential.MLP(vectorDims, forwardMLP_size, vectorDims, activation, $"forward-{i}"))];
            translationMLP = Sequential.MLP(vectorDims, translationMLP_size, tokenizer.tokenCount, activation, "translation");

            parameters = [.. positionalEncoding.GetParameters(), eigenVectors, .. keyMatrices, .. queryMatrices, .. valueMatrices, .. forwardMLPs.SelectMany(m => m.GetParameters()), ..translationMLP.GetParameters()];
        }

        public Matrix TokensToVectors(int[] tokens)
        {
            Matrix[] output = new Matrix[tokens.Length];

            for (int i = 0; i < tokens.Length; i++)
            {
                output[i] = eigenVectors[tokens[i]];
            }

            return concatenate.Apply(output);
        }

        public Matrix Forward(Matrix m)
        {
            m = positionalEncoding.Forward(m);

            for (int layer = 0; layer < layers; layer++)
            {
                {
                    Matrix n = normilization.Forward(m);

                    Matrix keys = keyMatrices[layer] << n;
                    Matrix queries = (queryMatrices[layer] << n).T();
                    Matrix values = valueMatrices[layer] << n;

                    Matrix preweights = queries << keys;

                    Matrix masked = maskNegInf.Apply(preweights);

                    Matrix weights = softMax.Forward(masked);

                    m += values << weights;
                }

                {
                    Matrix n = normilization.Forward(m);

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
