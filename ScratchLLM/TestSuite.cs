using System;
using System.Text;
using Duck;
using Duck.Functions.Basic;
using Duck.Functions.Value.Single;
using Duck.Modules.Basic;

public static partial class TestSuite
{
    public static void RunAllExpanded()
    {
        Console.OutputEncoding = Encoding.UTF8;

        // Core
        //TestTranspose();
        TestDoubleTranspose();
        TestIdentityBehavior();

        // Elementwise
        //TestElementwiseSin();
        TestElementwiseCos();
        TestElementwiseExpLog();

        // Shape / Structure
        TestShapePreservation();
        TestShapeEdgeCases();

        // Extend / Broadcast
        TestExtendColumn();
        TestExtendRow();
        TestBroadcastAdd();
        TestBroadcastMul();
        TestBroadcastMixedShapes();

        // Arithmetic
        TestAdd();
        TestSub();
        TestMulElementwise();
        TestDivElementwise();

        // Matrix multiplication
        TestMatMulSquare();
        TestMatMulRectA();
        TestMatMulRectB();
        TestMatMulChain();

        // Stability / Edge
        TestZeroMatrix();
        TestNegativeValues();
        TestLargeValues();

        Console.WriteLine("ALL EXPANDED TESTS PASSED");
    }

    #region HELPERS

    static void AssertEqual(Matrix a, Matrix b, float eps = 1e-4f)
    {
        if (a.shape != b.shape)
            throw new Exception($"Shape mismatch {a.shape} vs {b.shape}");

        for (int y = 0; y < a.shape.height; y++)
            for (int x = 0; x < a.shape.width; x++)
            {
                float av = a.values[x, y];
                float bv = b.values[x, y];

                if (MathF.Abs(av - bv) > eps)
                    throw new Exception($"Mismatch ({x},{y}) {av} != {bv}");
            }
    }

    static Matrix M(float[,] v) => new Matrix(v);

    #endregion

    // =========================================================
    // TRANSPOSE
    // =========================================================

    static void TestDoubleTranspose()
    {
        Matrix a = M(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        AssertEqual(a.T().T(), a);
        Console.WriteLine("Double Transpose PASS");
    }

    static void TestIdentityBehavior()
    {
        Matrix a = M(new float[,] { { 7 } });
        AssertEqual(a.T(), a);
        Console.WriteLine("Identity Transpose PASS");
    }

    // =========================================================
    // ELEMENTWISE
    // =========================================================

    static void TestElementwiseCos()
    {
        Matrix a = M(new float[,] { { 0, 3.1415926f } });
        Matrix r = new Apply<Cos>().Forward(a);

        Matrix e = M(new float[,] { { 1, -1 } });

        AssertEqual(r, e, 1e-3f);
        Console.WriteLine("Cos PASS");
    }

    static void TestElementwiseExpLog()
    {
        Matrix a = M(new float[,] { { 0, 1 } });

        Matrix exp = new Apply<Exp>().Forward(a);

        Matrix expected = M(new float[,] { { 1, MathF.E } });

        AssertEqual(exp, expected, 1e-3f);
        Console.WriteLine("Exp PASS");
    }

    // =========================================================
    // SHAPES
    // =========================================================

    static void TestShapePreservation()
    {
        Matrix a = M(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix b = new Apply<Sin>().Forward(a);

        if (a.shape != b.shape)
            throw new Exception("Shape not preserved");

        Console.WriteLine("Shape Preservation PASS");
    }

    static void TestShapeEdgeCases()
    {
        Matrix a = M(new float[,] { { 1 } });
        Matrix b = new Extend(FunctionType.Row).Apply((a, 5));

        if (b.shape.width != 5)
            throw new Exception("Extend row failed");

        Console.WriteLine("Shape Edge PASS");
    }

    // =========================================================
    // EXTEND
    // =========================================================

    static void TestExtendColumn()
    {
        Matrix a = M(new float[,] { { 1 }, { 2 } });
        Matrix r = new Extend(FunctionType.Column).Apply((a, 3));

        Matrix e = M(new float[,] {
            {1,1,1},
            {2,2,2}
        });

        AssertEqual(r, e);
        Console.WriteLine("Extend Column PASS");
    }

    static void TestExtendRow()
    {
        Matrix a = M(new float[,] { { 1, 2 } });
        Matrix r = new Extend(FunctionType.Row).Apply((a, 3));

        Matrix e = M(new float[,] {
            {1,2},
            {1,2},
            {1,2}
        });

        AssertEqual(r, e);
        Console.WriteLine("Extend Row PASS");
    }

    // =========================================================
    // BROADCASTING
    // =========================================================

    static void TestBroadcastAdd()
    {
        Matrix a = M(new float[,] { { 1, 2, 3 } });
        Matrix b = M(new float[,] { { 10 }, { 20 }, { 30 } });

        Matrix r = a + b;

        Matrix e = M(new float[,] {
            {11,12,13},
            {21,22,23},
            {31,32,33}
        });

        AssertEqual(r, e);
        Console.WriteLine("Broadcast Add PASS");
    }

    static void TestBroadcastMul()
    {
        Matrix a = M(new float[,] { { 2, 2 } });
        Matrix b = M(new float[,] { { 3 }, { 4 } });

        Matrix r = a * b;

        Matrix e = M(new float[,] {
            {6,6},
            {8,8}
        });

        AssertEqual(r, e);
        Console.WriteLine("Broadcast Mul PASS");
    }

    static void TestBroadcastMixedShapes()
    {
        Matrix a = M(new float[,] { { 1 } });
        Matrix b = M(new float[,] { { 10, 20, 30 } });

        Matrix r = a + b;

        Matrix e = M(new float[,] {
            {11,21,31}
        });

        AssertEqual(r, e);
        Console.WriteLine("Broadcast Mixed PASS");
    }

    // =========================================================
    // ARITHMETIC
    // =========================================================

    static void TestAdd()
    {
        Matrix a = M(new float[,] { { 1, 2 } });
        Matrix b = M(new float[,] { { 3, 4 } });

        AssertEqual(a + b, M(new float[,] { { 4, 6 } }));
        Console.WriteLine("Add PASS");
    }

    static void TestSub()
    {
        Matrix a = M(new float[,] { { 5, 5 } });
        Matrix b = M(new float[,] { { 2, 3 } });

        AssertEqual(a - b, M(new float[,] { { 3, 2 } }));
        Console.WriteLine("Sub PASS");
    }

    static void TestMulElementwise()
    {
        Matrix a = M(new float[,] { { 2, 3 } });
        Matrix b = M(new float[,] { { 4, 5 } });

        AssertEqual(a * b, M(new float[,] { { 8, 15 } }));
        Console.WriteLine("Elementwise Mul PASS");
    }

    static void TestDivElementwise()
    {
        Matrix a = M(new float[,] { { 10, 20 } });
        Matrix b = M(new float[,] { { 2, 4 } });

        AssertEqual(a / b, M(new float[,] { { 5, 5 } }));
        Console.WriteLine("Elementwise Div PASS");
    }

    // =========================================================
    // MATRIX MULTIPLICATION
    // =========================================================

    static void TestMatMulSquare()
    {
        Matrix a = M(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix b = M(new float[,] { { 5, 6 }, { 7, 8 } });

        AssertEqual(a >> b, M(new float[,] {
            {19,22},
            {43,50}
        }));

        Console.WriteLine("MatMul Square PASS");
    }

    static void TestMatMulRectA()
    {
        Matrix a = M(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        Matrix b = M(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });

        AssertEqual(a >> b, M(new float[,] {
            {22,28},
            {49,64}
        }));

        Console.WriteLine("MatMul Rect A PASS");
    }

    static void TestMatMulRectB()
    {
        Matrix a = M(new float[,] { { 1 }, { 2 }, { 3 } });
        Matrix b = M(new float[,] { { 4, 5, 6 } });

        AssertEqual(a >> b, M(new float[,] {
            {4,5,6},
            {8,10,12},
            {12,15,18}
        }));

        Console.WriteLine("MatMul Rect B PASS");
    }

    static void TestMatMulChain()
    {
        Matrix a = M(new float[,] { { 1, 2 } });
        Matrix b = M(new float[,] { { 3 }, { 4 } });
        Matrix c = M(new float[,] { { 5 } });

        Matrix r = ((a >> b) >> c);

        AssertEqual(r, M(new float[,] { { 55 } }));

        Console.WriteLine("MatMul Chain PASS");
    }

    // =========================================================
    // EDGE CASES
    // =========================================================

    static void TestZeroMatrix()
    {
        Matrix a = M(new float[,] { { 0, 0 }, { 0, 0 } });
        Matrix b = new Apply<Sin>().Forward(a);

        AssertEqual(b, a);
        Console.WriteLine("Zero Matrix PASS");
    }

    static void TestNegativeValues()
    {
        Matrix a = M(new float[,] { { -1, -2 } });
        Matrix b = new Apply<Abs>().Forward(a);

        AssertEqual(b, M(new float[,] { { 1, 2 } }));
        Console.WriteLine("Negative Values PASS");
    }

    static void TestLargeValues()
    {
        Matrix a = M(new float[,] { { 1e6f, -1e6f } });
        Matrix b = new Apply<Abs>().Forward(a);

        AssertEqual(b, M(new float[,] { { 1e6f, 1e6f } }));
        Console.WriteLine("Large Values PASS");
    }
}