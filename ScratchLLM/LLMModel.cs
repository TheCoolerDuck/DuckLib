
using Duck.CustomLLM.Library.Objects.MatrixObjects;
using Duck.Modules.Activation;
using Duck.Modules.Basic;
using Duck.Functions.Value.Single;
using Duck.Modules;
using Duck.Modules.Advanced;
using Duck.Tokenization;

namespace ScratchLLM
{
    internal class LLMModel : IModule
    {
        private const int layers = 8;
        private const int vectorDims = 16;
        private const int key_queryDims = 4;

        private readonly int[] key_queryMLP_size = [24, 16, 4];
        private readonly int[] valueMLP_size = [24, 16, 16];
        private readonly int[] forwardMLP_size = [16, 32];
        private readonly int[] translationMLP_size = [16, 32];

        private readonly IModule activation = new Swish();

        private readonly Matrix eigenVectors;
        private readonly Sequential keyMLP;
        private readonly Sequential queryMLP;
        private readonly Sequential valueMLP;
        private readonly Sequential forwardMLP;
        private readonly Sequential translationMLP;

        private readonly IModule normilization = new Apply<TanH>();

        private readonly BLOS positionalEncoding = new(vectorDims);

        private readonly Matrix[] parameters;

        public LLMModel(Tokenizer tokenizer)
        {
            eigenVectors = new Matrix(Matrix.Random(vectorDims, tokenizer.tokenCount));

            keyMLP = Sequential.MLP(layers + vectorDims, key_queryMLP_size, key_queryDims, activation);
            queryMLP = Sequential.MLP(layers + vectorDims, key_queryMLP_size, key_queryDims, activation);
            valueMLP = Sequential.MLP(layers + vectorDims, valueMLP_size, vectorDims, activation);

            forwardMLP = Sequential.MLP(layers + vectorDims, forwardMLP_size, vectorDims, activation);
            translationMLP = Sequential.MLP(vectorDims, translationMLP_size, tokenizer.tokenCount, activation);

            parameters = [eigenVectors, .. keyMLP.GetParameters(), .. queryMLP.GetParameters(), .. valueMLP.GetParameters(), .. forwardMLP.GetParameters(), ..positionalEncoding.GetParameters(), ..translationMLP.GetParameters()];
        }

        public Matrix TokensToVectors(int[] tokens)
        {
            Matrix output = eigenVectors[tokens[0]];

            for (int i = 1; i < tokens.Length; i++)
            {
                output &= eigenVectors[tokens[i]];
            }

            return output;
        }

        public Matrix Forward(Matrix m)
        {
            /*
            int sequenceLength = m.shape.width;
            Matrix tri = new Matrix(Matrix.Tri(sequenceLength), false).T();

            for (int layer = 0; layer < layers; layer++)
            {
                Matrix layerEncoding = new Matrix(Matrix.OneHot(layer, layers, sequenceLength), false).T();
                {
                    Matrix layerEncodedVectors = normilization.Forward(m) & layerEncoding.T();

                    Matrix keys = keyMLP.Forward(layerEncodedVectors);
                    Matrix queries = queryMLP.Forward(layerEncodedVectors).T();
                    Matrix values = valueMLP.Forward(layerEncodedVectors);

                    Matrix weights = (queries << keys) * tri;

                    m += values << weights;
                }

                {
                    Matrix layerEncodedVectors = normilization.Forward(m) & layerEncoding.T();

                    m += forwardMLP.Forward(layerEncodedVectors);
                }
            }

            return translationMLP.Forward(m);

            */
            return m;
        }

        public Matrix[] GetParameters()
        {
            return parameters;
        }
    }
}
