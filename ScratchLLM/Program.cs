
using Duck;
using Duck.Functional.Elementary;
using Duck.Functions.Parameters;
using Duck.Functions.Value.Double;
using Duck.Functions.Value.Single;
using Duck.Management;
using Duck.Modules.Basic;
using Duck.Modules.Loss;
using Duck.Modules.PositionalEncoding;
using Duck.Optimization;
using Duck.Tokenization;
using ScratchLLM;
using System;
using System.Diagnostics;
using System.Text;

Console.OutputEncoding = System.Text.Encoding.UTF8;

//DeviceManager.defaultDevice = Device.GPU;

//TestSuite.RunAllExpanded();

//return;

Tokenizer tokenizer = new("C:\\Users\\pjsol\\source\\repos\\DuckLib\\ScratchLLM\\Data\\TokenizationData4000.txt");

SimpleLLMModel model = new(tokenizer);

int[] data = tokenizer.tokenize(File.ReadAllText("C:\\Users\\pjsol\\source\\repos\\DuckLib\\ScratchLLM\\Data\\RickAndMorty.txt"));

Optimizer optimizer = new AdamW(model.GetParameters(), lr: 0.001f, weightDecay: 0.0001f);
CrossEntropy loss = new();

Random random = new();

Stopwatch watch = Stopwatch.StartNew();

const int sampleSize = 1027;

for (int i = 0; i < 250; i++)
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
    Matrix aL = new SymmetricApplication<Add>().Apply(l);
    Console.WriteLine($"{i}/{watch.ElapsedMilliseconds}/{aL.values[0, 0] / l.size}");
}

