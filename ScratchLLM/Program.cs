

using Duck.CustomLLM.Library.Objects.MatrixObjects;
using Duck.Device_Management;
using Duck.Functions.Basic;
using Duck.Functions.Parameters;
using Duck.Modules.Basic;
using Duck.Modules.Loss;
using Duck.Optimization;
using Duck.Tokenization;
using ScratchLLM;

Console.OutputEncoding = System.Text.Encoding.UTF8;

Reporter.settings.millisecondsBetweenReports = long.MaxValue;

DeviceManager.defaultDevice = Device.GPU;

Matrix x = new Matrix(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });
Matrix y = new Matrix(new float[,] { { 7, 8 }, { 9, 10 }, { 11, 12 } });
Matrix z = y << x;

Matrix a = new Matrix(Matrix.Ones(100, 1)) * 5;
Matrix b = new Matrix(Matrix.Ones(100, 1)) * 3;

Matrix c = a & b & a;


Console.WriteLine(c);

Console.WriteLine(c << z);

/*

Tokenizer tokenizer = new("C:\\Users\\pjsol\\source\\repos\\DuckLib\\ScratchLLM\\Data\\TokenizationData.txt");

DeviceManager.defaultDevice = Device.GPU;

SimpleLLMModel model = new(tokenizer);

Reporter.Update(new Reporter.Report("Tokenizing Data", "Started"));

int[] data = tokenizer.tokenize(File.ReadAllText("C:\\Users\\pjsol\\source\\repos\\DuckLib\\ScratchLLM\\Data\\RickAndMorty.txt"));

Reporter.Update(new Reporter.Report("Tokenizing Data", "Finished"));

Optimizer optimizer = new AdamW(model.GetParameters(), lr: 0.003);
ILoss<DoubleMatrix> loss = new CrossEntropy();

Random random = new();

Reporter.Publish();

const int sampleSize = 1027;

for (int i = 0; i < 10_000; i++)
{
    int j = random.Next(data.Length - sampleSize);

    int[] tokens = data[j..(j + sampleSize)];
    Matrix input = model.TokensToVectors(tokens[1..^2]);

    int[] truthTokens = tokens[2..^1];
    double[,] truthValues = new double[1, truthTokens.Length];

    for (int k = 0; k < truthTokens.Length; k++)
        truthValues[0, k] = truthTokens[k];

    Matrix truth = new(truthValues, false);

    Reporter.Update(new Reporter.Report("Model Forward", "Started"));

    Matrix result = model.Forward(input.T());
    Matrix l = loss.Apply((result, truth));
    Reporter.Update(new Reporter.Report("Model Forward", "Finished"));
    Reporter.Update(new Reporter.Report("Model Backwards", "Started"));
    Reporter.Publish();
    l.Backwards();
    Reporter.Update(new Reporter.Report("Model Backwards", "Finished"));
    Reporter.Update(new Reporter.Report("Model Optimization", "Started"));
    Reporter.Publish();
    optimizer.step();
    l.ZeroGradient();
    Matrix aL = new Mean(FunctionType.Whole).Forward(l);
    Reporter.Update(new Reporter.Report("Model Optimization", "Finished"));
    Reporter.Update(new Reporter.Report("Model Loss", aL.values[0, 0].ToString()));
    Reporter.Publish();
}

*/