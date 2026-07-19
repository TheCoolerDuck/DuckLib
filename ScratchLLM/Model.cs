
using Duck;
using Duck.Functional.Elementary.Mask;
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
    internal class Model : Module
    {
        private readonly int layers;
        private readonly int vectorDims;
        private readonly int key_queryDims;
        private readonly int[] forwardMLP_size;
        private readonly int[] translationMLP_size;

        private readonly Matrix eigenVectors;
        private readonly Matrix[] keyMatrices;
        private readonly Matrix[] queryMatrices;
        private readonly Matrix[] valueMatrices;
        private readonly Sequential[] forwardMLPs;
        private readonly Sequential translationMLP;

        private readonly Matrix LPE;

        private readonly SoftMax softMax;

        private readonly LinearTanh[] normilization;

        private readonly Matrix[] parameters;

        public Model(
            int tokenCount,
            int layers,
            int vectorDims,
            int key_queryDims,
            int[] forwardMLP_size,
            int[] translationMLP_size,
            Model? parent = null,
            string name = "Model") : base(parent, name)
        {
            this.layers = layers;
            this.vectorDims = vectorDims;
            this.key_queryDims = key_queryDims;
            this.forwardMLP_size = forwardMLP_size;
            this.translationMLP_size = translationMLP_size;

            softMax = new SoftMax(this);

            eigenVectors = new Matrix(Matrix.Random(tokenCount, vectorDims), new MatrixOptions() { Name = "eigen vectors"});

            LPE = new Matrix(Matrix.Random(1024, vectorDims), new MatrixOptions() { Name = "LPE" });

            normilization = [.. Enumerable.Range(0, layers * 2 + 1).Select(i => new LinearTanh(vectorDims, this, "linear Tanh " + i))];

            keyMatrices = [.. Enumerable.Range(0, layers).Select(i => new Matrix(Matrix.Random(vectorDims, key_queryDims) / MathF.Sqrt(vectorDims), new MatrixOptions() { Name = $"key-{i}" } ))]; 
            queryMatrices = [.. Enumerable.Range(0, layers).Select(i => new Matrix(Matrix.Random(vectorDims, key_queryDims) / MathF.Sqrt(vectorDims), new MatrixOptions() { Name = $"query-{i}" }))];
            valueMatrices = [.. Enumerable.Range(0, layers).Select(i => new Matrix(Matrix.Random(vectorDims, vectorDims) / MathF.Sqrt(vectorDims), new MatrixOptions() { Name = $"value-{i}" }))];

            {

                forwardMLPs = [.. Enumerable.Range(0, layers).Select(_ => {
                    List<Module> modules = [];

                    int inSize = vectorDims;
                    int i = 0;
                    foreach (int h in forwardMLP_size)
                    {
                        modules.Add(new SwiGLU(inSize, h, this, "Forward " + _ + " SwiGLU " + i++));
                        inSize = h;
                    }

                    modules.Add(new Linear(inSize, vectorDims, this));

                    return new Sequential([.. modules], this);
                })];
            }

            {
                List<Module> modules = [];

                int inSize = vectorDims;
                int i = 0;
                foreach (int h in translationMLP_size)
                {
                    modules.Add(new SwiGLU(inSize, h, this, "Translation SwiGLU " + i++));
                    inSize = h;
                }

                modules.Add(new Linear(inSize, tokenCount, this));

                translationMLP = new Sequential([.. modules], this);
            }

            parameters = [eigenVectors, LPE, ..normilization.SelectMany(m => m.GetParameters()), .. keyMatrices, .. queryMatrices, .. valueMatrices, .. forwardMLPs.SelectMany(m => m.GetParameters()), ..translationMLP.GetParameters()];
        }

        public Matrix TokensToVectors(int[] tokens)
        {
            return eigenVectors.GetRows(tokens);
        }

        public override Matrix Forward(Matrix m)
        {
            m += LPE;

            Mask mask = new(float.NegativeInfinity, Mask.TriL(m.shape.width));

            for (int layer = 0; layer < layers; layer++)
            {
                {
                    Matrix n = normilization[layer * 2].Forward(m);

                    Matrix keys = n >> keyMatrices[layer];
                    Matrix queries = n >> queryMatrices[layer];
                    Matrix values = n >> valueMatrices[layer];

                    Matrix preweights = (queries >> keys.T()) / MathF.Sqrt(key_queryDims);

                    Matrix masked = mask.Apply(preweights);

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

        public override Matrix[] GetParameters()
        {
            return parameters;
        }
    }
}
