
using System;
using System.Text;

using Duck;
using Duck.Functions.Basic;
using Duck.Functions.Parameters;
using Duck.Functions.Value.Single;
using Duck.Management;
using Duck.Modules.Basic;
using Duck.Modules.Loss;
using Duck.Optimization;
using Duck.Tokenization;
using ScratchLLM;

Console.OutputEncoding = System.Text.Encoding.UTF8;

DeviceManager.defaultDevice = Device.GPU;

TestSuite.RunAllExpanded();

Tokenizer tokenizer = new("C:\\Users\\pjsol\\source\\repos\\DuckLib\\ScratchLLM\\Data\\TokenizationData.txt");

SimpleLLMModel model = new(tokenizer);

int[] data = tokenizer.tokenize(File.ReadAllText("C:\\Users\\pjsol\\source\\repos\\DuckLib\\ScratchLLM\\Data\\RickAndMorty.txt"));

Optimizer optimizer = new AdamW(model.GetParameters(), lr: 0.003f);
CrossEntropy loss = new();

Random random = new();

Reporter.Publish();

const int sampleSize = 1027;

for (int i = 0; i < 10_000; i++)
{
    int j = random.Next(data.Length - sampleSize);

    int[] tokens = data[j..(j + sampleSize)];
    Matrix input = model.TokensToVectors(tokens[1..^2]);

    int[] truth = tokens[2..^1];

    Matrix result = model.Forward(input.T());
    Matrix l = loss.Apply((result, truth));
    l.Backwards();
    optimizer.step();
    l.ZeroGradient();
    Matrix aL = new Mean(FunctionType.Whole).Forward(l);
    Console.WriteLine($"Model Loss: {aL.values[0, 0]}");
}

