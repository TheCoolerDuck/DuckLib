using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Duck.Tokenization
{
    internal struct TokenMap(int a, int b, int c)
    {
        public int a = a;
        public int b = b;
        public int c = c;
    }


    public class Tokenizer
    {
        public int tokenCount => tokenMaps.Length + baseTokens.Length + specialTokens.Length;

        private readonly TokenMap[] tokenMaps;
        private readonly char[] baseTokens;
        private readonly string[] specialTokens;

        private readonly Regex regex = new(@"(\<\|[a-zA-z]+\|\>)|( ?(([a-zA-z]+('[st])?:?)|[1-9]+|.))");

        public Tokenizer(string textToLearn, int tokenCount, string[]? specialTokens = null)
        {
            this.specialTokens = specialTokens ?? [];
            baseTokens = gatherBaseTokens();
            List<List<int>> tokens = textToBaseTokens(textToLearn);

            int baseTokenCount = baseTokens.Length + this.specialTokens.Length;
            int tokensToAdd = tokenCount - baseTokenCount;

            tokenMaps = new TokenMap[tokensToAdd];

            for (int i = 0; i < tokensToAdd; i++)
            {
                Dictionary<(int, int), int> map = gatherTokenPairFrequency();
                (int, int) max = map.MaxBy(x => x.Value).Key;
                TokenMap tokenMap = new(max.Item1, max.Item2, i + baseTokenCount);
                tokenMaps[i] = tokenMap;
                mergePair(tokenMap, tokens);
            }

            char[] gatherBaseTokens()
            {
                List<char> chars = [];
                for (int i = 0; i < 256; i++)
                    chars.Add((char)i);
                foreach (char c in textToLearn)
                    if (!chars.Contains(c))
                        chars.Add(c);
                return [.. chars];
            }

            Dictionary<(int, int), int> gatherTokenPairFrequency()
            {
                Dictionary<(int, int), int> map = [];
                foreach (List<int> list in tokens)
                    for (int i = 0; i < list.Count - 1; i++)
                        map[(list[i], list[i + 1])] = map.GetValueOrDefault((list[i], list[i + 1]), 0) + 1;
                return map;
            }
        }

        public Tokenizer(string loadPath)
        {
            string text = File.ReadAllText(loadPath);
            int i = 0;

            int baseTokenCount = int.Parse(parseToNewLine());

            baseTokens = new char[baseTokenCount];

            for (int j = 0; j < baseTokenCount; j++)
                baseTokens[j] = text[i + j];

            i += baseTokenCount + 3;

            int specialTokenCount = int.Parse(parseToNewLine());

            specialTokens = new string[specialTokenCount];

            for (int j = 0; j < specialTokenCount; j++)
                specialTokens[j] = parseToNewLine();

            int mappingCount = int.Parse(parseToNewLine());

            i += 2;

            tokenMaps = new TokenMap[mappingCount];

            for (int j = 0; j < mappingCount; j++)
            {
                string line = parseToNewLine();
                int comma = line.IndexOf(',');
                int arrow = line.IndexOf('>');
                int a = int.Parse(line[..comma]);
                int b = int.Parse(line.Substring(comma + 1, arrow - comma - 1));
                int c = int.Parse(line.Substring(arrow + 1, line.Length - arrow - 1));

                tokenMaps[j] = new TokenMap(a, b, c);
            }

            string parseToNewLine()
            {
                StringBuilder sb = new();

                while (text[i] != '\n')
                    sb.Append(text[i++]);

                i++;

                return sb.ToString();
            }
        }

        private List<List<int>> textToBaseTokens(string text)
        {
            MatchCollection splitText = regex.Matches(text);
            List<List<int>> output = [];

            foreach (Match match in splitText)
            {
                if (!match.Success) continue;

                if (specialTokens != null && specialTokens.Contains(match.Value))
                {
                    output.Add([Array.IndexOf(specialTokens, match.Value) + baseTokens.Length]);
                }
                else
                {
                    output.Add([.. match.Value.Select(c => Array.IndexOf(baseTokens, c))]);
                }
            }
            return output;
        }

        private static void mergePair(TokenMap tokenMap, List<List<int>> tokens)
        {
            Parallel.ForEach(tokens, list =>
            {
                for (int i = 0; i < list.Count - 1; i++)
                {
                    if (list[i] == tokenMap.a && list[i + 1] == tokenMap.b)
                    {
                        list[i] = tokenMap.c;
                        list.RemoveAt(i + 1);
                    }
                }
            });
        }
        public string decodeToken(int token)
        {
            if (token == -1)
                return "%%TOKENERROR%%";

            if (token < baseTokens!.Length)
                return baseTokens[token].ToString();
            if (specialTokens != null && token < baseTokens!.Length + specialTokens.Length)
                return specialTokens![token - baseTokens!.Length];

            foreach (TokenMap map in tokenMaps)
            {   
                if (map.c == token)
                    return decodeToken(map.a) + decodeToken(map.b);
            }

            return "%%TOKENERROR%%";
        }
        public int[] tokenize(string text)
        {
            List<List<int>> tokens = textToBaseTokens(text);

            foreach(TokenMap map in tokenMaps)
                mergePair(map, tokens);

            return [.. tokens.SelectMany(l => l)];
        }
        public void save(string path)
        {
            StringBuilder sb = new();
            sb.AppendLine(baseTokens.Length + "\n" + string.Join("", baseTokens) + "\n");
            sb.AppendLine(specialTokens.Length + "\n" + string.Join("\n", specialTokens) + "\n" + tokenMaps.Length + "\n");

            foreach (TokenMap map in tokenMaps)
                sb.AppendLine($"{map.a},{map.b}>{map.c}");

            File.WriteAllText(path, sb.ToString());
        }
    }
}
