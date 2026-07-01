using System;

public class ValuePackage
{
    public readonly float[,] values;

    public int Width => values.GetLength(0);
    public int Height => values.GetLength(1);

    public ValuePackage(float[,] values)
    {
        this.values = values;
    }

    public float this[int x, int y] => values[x, y];

    // Matrix addition
    public static ValuePackage operator +(ValuePackage a, ValuePackage b)
    {
        CheckSameSize(a, b);

        float[,] result = new float[a.Width, a.Height];

        for (int i = 0; i < a.Width; i++)
            for (int j = 0; j < a.Height; j++)
                result[i, j] = a.values[i, j] + b.values[i, j];

        return new ValuePackage(result);
    }

    // Matrix subtraction
    public static ValuePackage operator -(ValuePackage a, ValuePackage b)
    {
        CheckSameSize(a, b);

        float[,] result = new float[a.Width, a.Height];

        for (int i = 0; i < a.Width; i++)
            for (int j = 0; j < a.Height; j++)
                result[i, j] = a.values[i, j] - b.values[i, j];

        return new ValuePackage(result);
    }

    // Matrix multiplication
    public static ValuePackage operator *(ValuePackage a, ValuePackage b)
    {
        if (a.Height != b.Width)
            throw new ArgumentException("Matrix dimensions do not match for multiplication.");

        float[,] result = new float[a.Width, b.Height];

        for (int i = 0; i < a.Width; i++)
        {
            for (int j = 0; j < b.Height; j++)
            {
                float sum = 0;

                for (int k = 0; k < a.Height; k++)
                    sum += a.values[i, k] * b.values[k, j];

                result[i, j] = sum;
            }
        }

        return new ValuePackage(result);
    }

    // Scalar multiplication
    public static ValuePackage operator *(ValuePackage a, float scalar)
    {
        float[,] result = new float[a.Width, a.Height];

        for (int i = 0; i < a.Width; i++)
            for (int j = 0; j < a.Height; j++)
                result[i, j] = a.values[i, j] * scalar;

        return new ValuePackage(result);
    }

    public static ValuePackage operator *(float scalar, ValuePackage a) => a * scalar;

    // Scalar division
    public static ValuePackage operator /(ValuePackage a, float scalar)
    {
        float[,] result = new float[a.Width, a.Height];

        for (int i = 0; i < a.Width; i++)
            for (int j = 0; j < a.Height; j++)
                result[i, j] = a.values[i, j] / scalar;

        return new ValuePackage(result);
    }

    // Unary negation
    public static ValuePackage operator -(ValuePackage a)
    {
        float[,] result = new float[a.Width, a.Height];

        for (int i = 0; i < a.Width; i++)
            for (int j = 0; j < a.Height; j++)
                result[i, j] = -a.values[i, j];

        return new ValuePackage(result);
    }

    // Transpose
    public ValuePackage Transpose()
    {
        float[,] result = new float[Height, Width];

        for (int i = 0; i < Width; i++)
            for (int j = 0; j < Height; j++)
                result[j, i] = values[i, j];

        return new ValuePackage(result);
    }

    private static void CheckSameSize(ValuePackage a, ValuePackage b)
    {
        if (a.Width != b.Width || a.Height != b.Height)
            throw new ArgumentException("Matrix dimensions must match.");
    }
}