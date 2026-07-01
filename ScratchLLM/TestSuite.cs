using System;
using System.Collections.Generic;
using System.Text;
using Duck;
using Duck.Functions.Basic;
using Duck.Functions.Value.Double;
using Duck.Functions.Value.Single;
using Duck.Modules.Basic;
using Duck.Modules.Activation;
using Duck.Modules.Loss;
using Duck.Modules.Normalization;
using Duck.Functional.Elementary;

// ─────────────────────────────────────────────────────────────────────────────
//  Duck — Comprehensive Test Suite
//  • Every test is isolated; a failure prints FAIL and continues (no throws)
//  • Final summary prints pass / fail counts per group and overall
// ─────────────────────────────────────────────────────────────────────────────

public static partial class TestSuite
{
    // =========================================================================
    //  RUNNER
    // =========================================================================

    public static void RunAllExpanded()
    {
        Console.OutputEncoding = Encoding.UTF8;
        _results.Clear();

        // ── Transpose: structural ──────────────────────────────────────────────
        Run("Transpose / Double Transpose", TestDoubleTranspose);
        Run("Transpose / Single Element", TestTransposeSingleElement);
        Run("Transpose / Dimensions Flipped", TestTransposeDimensions);
        Run("Transpose / Non-Square Wide (1×4→4×1)", TestTransposeNonSquareWide);
        Run("Transpose / Non-Square Tall (4×1→1×4)", TestTransposeNonSquareTall);
        Run("Transpose / Column Vector", TestTransposeColumnVector);
        Run("Transpose / Row Vector", TestTransposeRowVector);
        Run("Transpose / Value Mapping a[i,j]==t[j,i]", TestTransposePreservesValues);
        Run("Transpose / Square 3×3", TestTranspose3x3);
        Run("Transpose / 5-Element Row", TestTranspose5ElementRow);
        Run("Transpose / With Zeros", TestTransposeWithZeros);
        Run("Transpose / With Negative Values", TestTransposeWithNegativeValues);
        Run("Transpose / With Large Values", TestTransposeWithLargeValues);
        Run("Transpose / With Mixed Sign Values", TestTransposeWithMixedSigns);

        // ── Transpose: algebraic identities ───────────────────────────────────
        Run("Transpose-Algebra / (A^T)^T == A", TestDoubleTransposeIdentity);
        Run("Transpose-Algebra / Triple Transpose", TestTripleTranspose);
        Run("Transpose-Algebra / (A+B)^T == A^T+B^T", TestTransposeDistributesOverAdd);
        Run("Transpose-Algebra / (A-B)^T == A^T-B^T", TestTransposeDistributesOverSub);
        Run("Transpose-Algebra / (cA)^T == cA^T", TestTransposeScalarMul);
        Run("Transpose-Algebra / (AB)^T == B^T A^T", TestTransposeMatMulReversal);
        Run("Transpose-Algebra / A-A^T skew-symmetric", TestSubTransposedSkewSymmetric);
        Run("Transpose-Algebra / A+A^T symmetric", TestAddTransposedSymmetric);

        // ── Transpose: interaction with other ops ─────────────────────────────
        Run("Transpose-Ops / T then GetRows", TestTransposeThenGetRows);
        //Run("Transpose-Ops / T then GetColumns", TestTransposeThenGetColumns);
        Run("Transpose-Ops / Extend then T", TestExtendThenTranspose);
        Run("Transpose-Ops / T then Extend", TestTransposeThenExtend);
        Run("Transpose-Ops / Broadcast Add via T", TestTransposeBroadcastIdentity);
        Run("Transpose-Ops / f(A)^T == f(A^T) Sin", TestElementwiseSinCommuteT);
        Run("Transpose-Ops / f(A)^T == f(A^T) Cos", TestElementwiseCosCommuteT);
        Run("Transpose-Ops / f(A)^T == f(A^T) Exp", TestElementwiseExpCommuteT);
        Run("Transpose-Ops / f(A)^T == f(A^T) Abs", TestElementwiseAbsCommuteT);
        Run("Transpose-Ops / f(A)^T == f(A^T) Tanh", TestElementwiseTanhCommuteT);
        Run("Transpose-Ops / MatMul A^T A", TestMatMulATA);
        Run("Transpose-Ops / MatMul A A^T", TestMatMulAAT);

        // ── Elementwise single-value functions ────────────────────────────────
        Run("Elementwise / Sin(0)==0, Sin(π/2)==1", TestSin);
        Run("Elementwise / Cos(0)==1, Cos(π)==-1", TestCos);
        Run("Elementwise / Exp(0)==1, Exp(1)==e", TestExp);
        Run("Elementwise / Log(1)==0, Log(e)==1", TestLog);
        Run("Elementwise / Abs positive", TestAbsPositive);
        Run("Elementwise / Abs negative", TestAbsNegative);
        Run("Elementwise / Abs zero", TestAbsZero);
        Run("Elementwise / Tan(0)==0, Tan(π/4)≈1", TestTan);
        Run("Elementwise / Tanh(0)==0, Tanh large≈1", TestTanh);
        Run("Elementwise / Shape preserved by Sin", TestElementwiseShapePreservation);
        Run("Elementwise / Sin on zero matrix", TestSinZeroMatrix);
        Run("Elementwise / Exp then Log round-trip", TestExpLogRoundTrip);
        Run("Elementwise / Abs idempotent", TestAbsIdempotent);
        Run("Elementwise / Cos on transposed", TestCosOnTransposed);

        // ── Elementwise double-value functions ────────────────────────────────
        Run("Elementwise-D / Add", TestDVAdd);
        Run("Elementwise-D / Sub", TestDVSub);
        Run("Elementwise-D / Mul", TestDVMul);
        Run("Elementwise-D / Div", TestDVDiv);
        Run("Elementwise-D / Pow", TestDVPow);
        Run("Elementwise-D / Max", TestDVMax);
        Run("Elementwise-D / Min", TestDVMin);

        // ── Arithmetic operators (operator overloads) ─────────────────────────
        Run("Arithmetic / Add same shape", TestAdd);
        Run("Arithmetic / Sub same shape", TestSub);
        Run("Arithmetic / Mul elementwise", TestMulElementwise);
        Run("Arithmetic / Div elementwise", TestDivElementwise);
        Run("Arithmetic / Add negative values cancel", TestAddNegativeCancel);
        Run("Arithmetic / Sub result negative", TestSubResultNegative);
        Run("Arithmetic / Mul by zero", TestMulByZero);
        Run("Arithmetic / Div by one identity", TestDivByOne);
        Run("Arithmetic / A+A == 2A", TestAddSelf);
        Run("Arithmetic / A-A == 0", TestSubSelf);
        Run("Arithmetic / A*A elementwise square", TestMulSelf);
        Run("Arithmetic / A/A elementwise ones", TestDivSelf);

        // ── Broadcast ─────────────────────────────────────────────────────────
        Run("Broadcast / Scalar + Row", TestBroadcastScalarPlusRow);
        Run("Broadcast / Scalar * Matrix", TestBroadcastScalarMulMatrix);
        Run("Broadcast / Col + Row (outer-like)", TestBroadcastColPlusRow);
        Run("Broadcast / Col * Row", TestBroadcastColMulRow);
        Run("Broadcast / 1×1 + 1×3", TestBroadcastOneByOneRow);
        Run("Broadcast / Col vector 3×1 + scalar", TestBroadcastColPlusScalar);

        // ── Extend ────────────────────────────────────────────────────────────
        Run("Extend / Column ×3", TestExtendColumn);
        Run("Extend / Row ×3", TestExtendRow);
        Run("Extend / Column ×1 identity", TestExtendColumnByOne);
        Run("Extend / Row ×1 identity", TestExtendRowByOne);
        Run("Extend / Scalar column", TestExtendScalarColumn);
        Run("Extend / Scalar row", TestExtendScalarRow);

        // ── GetVectors (GetRows / GetColumns) ─────────────────────────────────
        Run("GetVectors / GetRows [2,0]", TestGetRowsBasic);
        Run("GetVectors / GetColumns [2,0]", TestGetColumnsBasic);
        Run("GetVectors / GetRows duplicates", TestGetRowsDuplicates);
        Run("GetVectors / GetColumns duplicates", TestGetColumnsDuplicates);
        Run("GetVectors / GetRows single", TestGetRowsSingle);
        //Run("GetVectors / GetColumns single", TestGetColumnsSingle);
        Run("GetVectors / GetRows all in order", TestGetRowsAllInOrder);
        Run("GetVectors / GetColumns all in order", TestGetColumnsAllInOrder);
        Run("GetVectors / GetRows reverse", TestGetRowsReverse);
        Run("GetVectors / GetColumns reverse", TestGetColumnsReverse);
        Run("GetVectors / GetRows on transposed", TestGetRowsOnTransposed);
        //Run("GetVectors / GetColumns on transposed", TestGetColumnsOnTransposed);
        Run("GetVectors / GetRows after add", TestGetRowsAfterAdd);
        //Run("GetVectors / GetColumns after mul", TestGetColumnsAfterMul);

        // ── Concatenate ───────────────────────────────────────────────────────
        Run("Concatenate / Horizontal 2 matrices", TestConcatHorizontal);
        Run("Concatenate / Vertical 2 matrices", TestConcatVertical);
        Run("Concatenate / Single row concat", TestConcatSingleRow);
        Run("Concatenate / Concat then T", TestConcatThenTranspose);


        // ── Sum ──────────────────────────────────────────────────────────────
        Run("Sum / Row-wise sum", TestSum);

        // ── Matrix Multiplication ─────────────────────────────────────────────
        Run("MatMul / Square 2×2", TestMatMulSquare);
        Run("MatMul / Rect A (2×3 >> 3×2)", TestMatMulRectA);
        Run("MatMul / Rect B col>>row outer product", TestMatMulRectB);
        Run("MatMul / Chain (AB)C", TestMatMulChain);
        Run("MatMul / Identity left", TestMatMulIdentityLeft);
        Run("MatMul / Identity right", TestMatMulIdentityRight);
        Run("MatMul / Zero left", TestMatMulZeroLeft);
        Run("MatMul / Zero right", TestMatMulZeroRight);
        Run("MatMul / 1×1 scalars", TestMatMulScalars);
        Run("MatMul / Dot product (inner)", TestMatMulDotProduct);
        Run("MatMul / Outer product", TestMatMulOuterProduct);
        Run("MatMul / 3×3 square", TestMatMulSquare3x3);
        Run("MatMul / Associativity (AB)C=A(BC)", TestMatMulAssociative);
        Run("MatMul / (AB)^T == B^T A^T", TestMatMulTransposeReversal);
        Run("MatMul / A^T A Gram matrix", TestMatMulGram);

        // ── Modules: Apply ────────────────────────────────────────────────────
        Run("Module-Apply / Sin", TestApplySin);
        Run("Module-Apply / Cos", TestApplyCos);
        Run("Module-Apply / Exp", TestApplyExp);
        Run("Module-Apply / Log", TestApplyLog);
        Run("Module-Apply / Abs", TestApplyAbs);
        Run("Module-Apply / Tanh", TestApplyTanh);
        Run("Module-Apply / Tan", TestApplyTan);

        // ── Modules: SoftMax ──────────────────────────────────────────────────
        Run("Module-SoftMax / Output sums to 1", TestSoftMaxSumsToOne);
        Run("Module-SoftMax / All equal inputs", TestSoftMaxEqualInputs);
        Run("Module-SoftMax / All outputs in (0,1)", TestSoftMaxRange);
        Run("Module-SoftMax / Largest input dominates", TestSoftMaxDominant);

        // ── Modules: Activation ───────────────────────────────────────────────
        Run("Activation / Sigmoid(0)==0.5", TestSigmoidAtZero);
        Run("Activation / Sigmoid large≈1", TestSigmoidLarge);
        Run("Activation / Sigmoid negative≈0", TestSigmoidNegative);
        Run("Activation / Swish(0)==0", TestSwishAtZero);

        // ── Stability / edge cases ────────────────────────────────────────────
        Run("Edge / All-zeros matrix", TestAllZeros);
        Run("Edge / All-ones MatMul", TestAllOnesMatMul);
        Run("Edge / All-negones MatMul", TestAllNegOnesMatMul);
        Run("Edge / 1×1 all ops", TestScalarAllOps);
        Run("Edge / Large values Abs", TestLargeValuesAbs);
        Run("Edge / Large MatMul", TestLargeMatMul);

        // ── Summary ───────────────────────────────────────────────────────────
        PrintSummary();
    }

    // =========================================================================
    //  SOFT-FAIL INFRASTRUCTURE
    // =========================================================================

    private record TestResult(string Name, bool Passed, string? Reason);
    private static readonly List<TestResult> _results = new();

    private static void Run(string name, Action test)
    {

        try
        {
            test();
            _results.Add(new TestResult(name, true, null));
            Console.WriteLine($"  ✓  {name}");
        }
        catch (Exception ex)
        {
            _results.Add(new TestResult(name, false, ex.Message));
            Console.WriteLine($"  ✗  {name}");
            Console.WriteLine($"       → {ex.Message}");
        }
    }

    private static void PrintSummary()
    {
        Console.WriteLine();
        Console.WriteLine(new string('═', 72));
        Console.WriteLine("  RESULTS BY GROUP");
        Console.WriteLine(new string('─', 72));

        string? lastGroup = null;
        int groupPass = 0, groupFail = 0;

        void FlushGroup()
        {
            if (lastGroup is null) return;
            string tag = groupFail == 0 ? "ALL PASS" : $"{groupFail} FAILED";
            Console.WriteLine($"    {lastGroup,-40} {groupPass,3} pass  {groupFail,3} fail  [{tag}]");
        }

        foreach (var r in _results)
        {
            string group = r.Name.Split('/')[0].Trim();
            if (group != lastGroup)
            {
                FlushGroup();
                lastGroup = group;
                groupPass = 0;
                groupFail = 0;
            }
            if (r.Passed) groupPass++; else groupFail++;
        }
        FlushGroup();

        int totalPass = 0, totalFail = 0;
        foreach (var r in _results) { if (r.Passed) totalPass++; else totalFail++; }

        Console.WriteLine(new string('─', 72));
        Console.WriteLine($"  TOTAL  {totalPass} passed,  {totalFail} failed  out of {_results.Count}");
        Console.WriteLine(new string('═', 72));

        if (totalFail > 0)
        {
            Console.WriteLine();
            Console.WriteLine("FAILED TESTS:");
            foreach (var r in _results)
                if (!r.Passed)
                    Console.WriteLine($"  • {r.Name}\n      {r.Reason}");
        }
    }

    // =========================================================================
    //  HELPERS
    // =========================================================================

    static Matrix M(float[,] v) => new Matrix(v);
    static Matrix Scalar(float v) => M(new float[,] { { v } });

    static void AssertEqual(Matrix a, Matrix b, float eps = 1e-4f)
    {
        if (a.shape != b.shape)
            throw new Exception($"Shape mismatch: {a.shape} vs {b.shape}");

        for (int y = 0; y < a.shape.height; y++)
            for (int x = 0; x < a.shape.width; x++)
            {
                float av = a.values[x, y];
                float bv = b.values[x, y];
                if (MathF.Abs(av - bv) > eps)
                    throw new Exception($"Value mismatch at ({x},{y}): got {av}, expected {bv}  (eps={eps})");
            }
    }

    static void AssertShape(Matrix m, int expectedWidth, int expectedHeight)
    {
        if (m.shape.height != expectedHeight || m.shape.width != expectedWidth)
            throw new Exception($"Shape wrong: got {m.shape}, expected ({expectedWidth}×{expectedHeight})");
    }

    static void AssertTrue(bool cond, string msg)
    {
        if (!cond) throw new Exception(msg);
    }

    static float Val(Matrix m, int x, int y) => m.transposed ? m.values[x, y] : m.values[y, x];

    // =========================================================================
    //  TRANSPOSE — STRUCTURAL
    // =========================================================================

    static void TestDoubleTranspose()
    {
        Matrix a = M(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        AssertEqual(a.T().T(), a);
    }

    static void TestTransposeSingleElement()
    {
        AssertEqual(Scalar(42).T(), Scalar(42));
    }

    static void TestTransposeDimensions()
    {
        Matrix a = M(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } }); // 2 rows, 3 cols
        Matrix t = a.T();
        AssertShape(t, expectedHeight: a.shape.width, expectedWidth: a.shape.height);
    }

    static void TestTransposeNonSquareWide()
    {
        Matrix a = M(new float[,] { { 10, 20, 30, 40 } });
        AssertEqual(a.T(), M(new float[,] { { 10 }, { 20 }, { 30 }, { 40 } }));
    }

    static void TestTransposeNonSquareTall()
    {
        Matrix a = M(new float[,] { { 10 }, { 20 }, { 30 }, { 40 } });
        AssertEqual(a.T(), M(new float[,] { { 10, 20, 30, 40 } }));
    }

    static void TestTransposeColumnVector()
    {
        Matrix a = M(new float[,] { { 1 }, { 2 }, { 3 } });
        AssertEqual(a.T(), M(new float[,] { { 1, 2, 3 } }));
    }

    static void TestTransposeRowVector()
    {
        Matrix a = M(new float[,] { { 1, 2, 3 } });
        AssertEqual(a.T(), M(new float[,] { { 1 }, { 2 }, { 3 } }));
    }

    static void TestTransposePreservesValues()
    {
        Matrix a = M(new float[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
        Matrix t = a.T();
        for (int y = 0; y < a.shape.height; y++)
            for (int x = 0; x < a.shape.width; x++)
                if (a.values[x, y] != t.values[y, x])
                    throw new Exception($"a[{x},{y}]={a.values[x, y]} but t[{y},{x}]={t.values[y, x]}");
    }

    static void TestTranspose3x3()
    {
        Matrix a = M(new float[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
        Matrix e = M(new float[,] { { 1, 4, 7 }, { 2, 5, 8 }, { 3, 6, 9 } });
        AssertEqual(a.T(), e);
    }

    static void TestTranspose5ElementRow()
    {
        Matrix a = M(new float[,] { { 1, 2, 3, 4, 5 } });
        Matrix e = M(new float[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
        AssertEqual(a.T(), e);
    }

    static void TestTransposeWithZeros()
    {
        Matrix a = M(new float[,] { { 0, 1 }, { 2, 0 } });
        Matrix e = M(new float[,] { { 0, 2 }, { 1, 0 } });
        AssertEqual(a.T(), e);
    }

    static void TestTransposeWithNegativeValues()
    {
        Matrix a = M(new float[,] { { -1, -2 }, { -3, -4 } });
        Matrix e = M(new float[,] { { -1, -3 }, { -2, -4 } });
        AssertEqual(a.T(), e);
    }

    static void TestTransposeWithLargeValues()
    {
        Matrix a = M(new float[,] { { 1e7f, -1e7f }, { 2e7f, -2e7f } });
        Matrix e = M(new float[,] { { 1e7f, 2e7f }, { -1e7f, -2e7f } });
        AssertEqual(a.T(), e, 1f);
    }

    static void TestTransposeWithMixedSigns()
    {
        Matrix a = M(new float[,] { { -5, 0, 5 }, { 3, -3, 0 } });
        Matrix e = M(new float[,] { { -5, 3 }, { 0, -3 }, { 5, 0 } });
        AssertEqual(a.T(), e);
    }

    // =========================================================================
    //  TRANSPOSE — ALGEBRAIC IDENTITIES
    // =========================================================================

    static void TestDoubleTransposeIdentity()
    {
        Matrix a = M(new float[,] { { 7, -2 }, { 0, 4 } });
        AssertEqual(a.T().T(), a);
    }

    static void TestTripleTranspose()
    {
        // ((A^T)^T)^T == A^T
        Matrix a = M(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        AssertEqual(a.T().T().T(), a.T());
    }

    static void TestTransposeDistributesOverAdd()
    {
        // (A+B)^T == A^T + B^T
        Matrix a = M(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix b = M(new float[,] { { 5, 6 }, { 7, 8 } });
        AssertEqual((a + b).T(), a.T() + b.T());
    }

    static void TestTransposeDistributesOverSub()
    {
        // (A-B)^T == A^T - B^T
        Matrix a = M(new float[,] { { 10, 20 }, { 30, 40 } });
        Matrix b = M(new float[,] { { 1, 2 }, { 3, 4 } });
        AssertEqual((a - b).T(), a.T() - b.T());
    }

    static void TestTransposeScalarMul()
    {
        // (s*A)^T == s * A^T   (broadcast scalar)
        Matrix a = M(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix s = Scalar(3);
        AssertEqual((a * s).T(), a.T() * s);
    }

    static void TestTransposeMatMulReversal()
    {
        // (AB)^T == B^T A^T
        Matrix a = M(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }); // 2×3
        Matrix b = M(new float[,] { { 7, 8, 9 }, { 10, 11, 12 } });   // 3×2
        AssertEqual((b >> a).T(), a.T() >> b.T(), 1e-3f);
    }

    static void TestSubTransposedSkewSymmetric()
    {
        // A - A^T: diagonal must be 0, off-diagonal must be antisymmetric
        Matrix a = M(new float[,] { { 1, 3 }, { 7, 2 } });
        Matrix r = a - a.T();
        AssertTrue(MathF.Abs(Val(r, 0, 0)) < 1e-4f, "Diagonal [0,0] not zero");
        AssertTrue(MathF.Abs(Val(r, 1, 1)) < 1e-4f, "Diagonal [1,1] not zero");
        AssertTrue(MathF.Abs(Val(r, 0, 1) + Val(r, 1, 0)) < 1e-4f, "Off-diagonal not antisymmetric");
    }

    static void TestAddTransposedSymmetric()
    {
        // A + A^T is symmetric: result[i,j] == result[j,i]
        Matrix a = M(new float[,] { { 1, 3 }, { 7, 2 } });
        Matrix s = a + a.T();
        AssertTrue(MathF.Abs(Val(s, 0, 1) - Val(s, 1, 0)) < 1e-4f, "Not symmetric");
    }

    // =========================================================================
    //  TRANSPOSE — INTERACTION WITH OTHER OPS
    // =========================================================================

    static void TestTransposeThenGetRows()
    {
        // Row i of A^T == Column i of A
        Matrix a = M(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } }); // 2×3 → A^T is 3×2
        // Row 1 of A^T should equal [2, 5]
        Matrix r = a.T().GetRows(new[] { 1 });
        Matrix e = M(new float[,] { { 2 }, { 5 } });
        AssertEqual(r, e);
    }

    static void TestTransposeThenGetColumns()
    {
        // Column j of A^T == Row j of A
        Matrix a = M(new float[,] { { 10, 20, 30 }, { 40, 50, 60 } }); // 2×3
        // Column 0 of A^T (3×2) should be [10, 20, 30]
        Matrix r = a.T().GetColumns(new[] { 0 });
        Matrix e = M(new float[,] { { 10 }, { 20 }, { 30 } });
        AssertEqual(r, e);
    }

    static void TestExtendThenTranspose()
    {
        Matrix a = M(new float[,] { { 1, 2 } }); // 1×2
        Matrix r = new Extend(FunctionType.Row).Apply((a, 3)); // 3×2
        Matrix e = M(new float[,] { { 1, 1, 1 }, { 2, 2, 2 } }); // 2×3
        AssertEqual(r.T(), e);
    }

    static void TestTransposeThenExtend()
    {
        Matrix a = M(new float[,] { { 1 }, { 2 } }); // 2×1
        Matrix t = a.T();                              // 1×2
        Matrix r = new Extend(FunctionType.Row).Apply((t, 3)); // 3×2
        Matrix e = M(new float[,] { { 1, 2 }, { 1, 2 }, { 1, 2 } });
        AssertEqual(r, e);
    }

    static void TestTransposeBroadcastIdentity()
    {
        // (s + A)^T == s + A^T   where s is scalar
        Matrix a = M(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix s = Scalar(10);
        AssertEqual((s + a).T(), s + a.T());
    }

    static void TestElementwiseSinCommuteT()
    {
        Matrix a = M(new float[,] { { 0f, MathF.PI / 2 }, { MathF.PI, 3 * MathF.PI / 2 } });
        AssertEqual(new Apply<Sin>().Forward(a).T(), new Apply<Sin>().Forward(a.T()), 1e-3f);
    }

    static void TestElementwiseCosCommuteT()
    {
        Matrix a = M(new float[,] { { 0f, MathF.PI }, { MathF.PI / 2, 3 * MathF.PI / 2 } });
        AssertEqual(new Apply<Cos>().Forward(a).T(), new Apply<Cos>().Forward(a.T()), 1e-3f);
    }

    static void TestElementwiseExpCommuteT()
    {
        Matrix a = M(new float[,] { { 0, 1 }, { 2, 3 } });
        AssertEqual(new Apply<Exp>().Forward(a).T(), new Apply<Exp>().Forward(a.T()), 1e-3f);
    }

    static void TestElementwiseAbsCommuteT()
    {
        Matrix a = M(new float[,] { { -1, 2 }, { -3, 4 } });
        AssertEqual(new Apply<Abs>().Forward(a).T(), new Apply<Abs>().Forward(a.T()));
    }

    static void TestElementwiseTanhCommuteT()
    {
        Matrix a = M(new float[,] { { -1, 0 }, { 1, 2 } });
        AssertEqual(new Apply<TanH>().Forward(a).T(), new Apply<TanH>().Forward(a.T()), 1e-4f);
    }

    static void TestMatMulATA()
    {
        // A^T >> A is always square and symmetric
        Matrix a = M(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }); // 3×2
        Matrix r = a.T() >> a; // 2×2
        AssertShape(r, 2, 2);
        // Must be symmetric
        AssertTrue(MathF.Abs(Val(r, 0, 1) - Val(r, 1, 0)) < 1e-3f, "A^T A not symmetric");
    }

    static void TestMatMulAAT()
    {
        // A >> A^T is always square and symmetric
        Matrix a = M(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } }); // 2×3
        Matrix r = a >> a.T(); // 2×2
        AssertShape(r, 2, 2);
        AssertTrue(MathF.Abs(Val(r, 0, 1) - Val(r, 1, 0)) < 1e-3f, "A A^T not symmetric");
    }

    // =========================================================================
    //  ELEMENTWISE — SINGLE-VALUE FUNCTIONS
    // =========================================================================

    static void TestSin()
    {
        Matrix a = M(new float[,] { { 0, MathF.PI / 2 } });
        AssertEqual(new Apply<Sin>().Forward(a), M(new float[,] { { 0, 1 } }), 1e-3f);
    }

    static void TestCos()
    {
        Matrix a = M(new float[,] { { 0, MathF.PI } });
        AssertEqual(new Apply<Cos>().Forward(a), M(new float[,] { { 1, -1 } }), 1e-3f);
    }

    static void TestExp()
    {
        Matrix a = M(new float[,] { { 0, 1 } });
        AssertEqual(new Apply<Exp>().Forward(a), M(new float[,] { { 1, MathF.E } }), 1e-3f);
    }

    static void TestLog()
    {
        Matrix a = M(new float[,] { { 1, MathF.E } });
        AssertEqual(new Apply<Log>().Forward(a), M(new float[,] { { 0, 1 } }), 1e-3f);
    }

    static void TestAbsPositive()
    {
        Matrix a = M(new float[,] { { 1, 2, 3 } });
        AssertEqual(new Apply<Abs>().Forward(a), a);
    }

    static void TestAbsNegative()
    {
        Matrix a = M(new float[,] { { -1, -2, -3 } });
        AssertEqual(new Apply<Abs>().Forward(a), M(new float[,] { { 1, 2, 3 } }));
    }

    static void TestAbsZero()
    {
        AssertEqual(new Apply<Abs>().Forward(Scalar(0)), Scalar(0));
    }

    static void TestTan()
    {
        Matrix a = M(new float[,] { { 0, MathF.PI / 4 } });
        AssertEqual(new Apply<Tan>().Forward(a), M(new float[,] { { 0, 1 } }), 1e-3f);
    }

    static void TestTanh()
    {
        Matrix a = M(new float[,] { { 0, 10f } });
        Matrix r = new Apply<TanH>().Forward(a);
        AssertTrue(MathF.Abs(Val(r, 0, 0)) < 1e-4f, "tanh(0) should be 0");
        AssertTrue(MathF.Abs(Val(r, 1, 0) - 1f) < 1e-3f, "tanh(10) should be ≈1");
    }

    static void TestElementwiseShapePreservation()
    {
        Matrix a = M(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix b = new Apply<Sin>().Forward(a);
        AssertShape(b, a.shape.height, a.shape.width);
    }

    static void TestSinZeroMatrix()
    {
        Matrix a = M(new float[,] { { 0, 0 }, { 0, 0 } });
        AssertEqual(new Apply<Sin>().Forward(a), a);
    }

    static void TestExpLogRoundTrip()
    {
        // log(exp(x)) ≈ x  for reasonable x
        Matrix a = M(new float[,] { { 1, 2, 3 } });
        Matrix r = new Apply<Log>().Forward(new Apply<Exp>().Forward(a));
        AssertEqual(r, a, 1e-3f);
    }

    static void TestAbsIdempotent()
    {
        // ||-x|| == |x|
        Matrix a = M(new float[,] { { -5, 3 } });
        Matrix r1 = new Apply<Abs>().Forward(a);
        Matrix r2 = new Apply<Abs>().Forward(r1);
        AssertEqual(r1, r2);
    }

    static void TestCosOnTransposed()
    {
        Matrix a = M(new float[,] { { 0, MathF.PI }, { MathF.PI / 2, 0 } });
        AssertEqual(new Apply<Cos>().Forward(a.T()), new Apply<Cos>().Forward(a).T(), 1e-3f);
    }

    // =========================================================================
    //  ELEMENTWISE — DOUBLE-VALUE FUNCTIONS
    //  These wrap the Duck.Functions.Value.Double types directly
    // =========================================================================

    static void TestDVAdd()
    {
        // Use the MatrixFunction wrapper for double-value functions
        Matrix a = M(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix b = M(new float[,] { { 10, 20 }, { 30, 40 } });
        Matrix r = new MatrixFunction<Add>().Apply((a, b));
        AssertEqual(r, M(new float[,] { { 11, 22 }, { 33, 44 } }));
    }

    static void TestDVSub()
    {
        Matrix a = M(new float[,] { { 5, 6 } });
        Matrix b = M(new float[,] { { 3, 4 } });
        AssertEqual(new MatrixFunction<Sub>().Apply((a, b)), M(new float[,] { { 2, 2 } }));
    }

    static void TestDVMul()
    {
        Matrix a = M(new float[,] { { 2, 3 } });
        Matrix b = M(new float[,] { { 4, 5 } });
        AssertEqual(new MatrixFunction<Mul>().Apply((a, b)), M(new float[,] { { 8, 15 } }));
    }

    static void TestDVDiv()
    {
        Matrix a = M(new float[,] { { 10, 20 } });
        Matrix b = M(new float[,] { { 2, 4 } });
        AssertEqual(new MatrixFunction<Div>().Apply((a, b)), M(new float[,] { { 5, 5 } }));
    }

    static void TestDVPow()
    {
        Matrix a = M(new float[,] { { 2, 3 } });
        Matrix b = M(new float[,] { { 3, 2 } });
        AssertEqual(new MatrixFunction<Pow>().Apply((a, b)), M(new float[,] { { 8, 9 } }), 1e-3f);
    }

    static void TestDVMax()
    {
        Matrix a = M(new float[,] { { 1, 5 } });
        Matrix b = M(new float[,] { { 3, 2 } });
        AssertEqual(new MatrixFunction<Max>().Apply((a, b)), M(new float[,] { { 3, 5 } }));
    }

    static void TestDVMin()
    {
        Matrix a = M(new float[,] { { 1, 5 } });
        Matrix b = M(new float[,] { { 3, 2 } });
        AssertEqual(new MatrixFunction<Min>().Apply((a, b)), M(new float[,] { { 1, 2 } }));
    }

    // =========================================================================
    //  ARITHMETIC OPERATORS
    // =========================================================================

    static void TestAdd()
    {
        AssertEqual(
            M(new float[,] { { 1, 2 } }) + M(new float[,] { { 3, 4 } }),
            M(new float[,] { { 4, 6 } }));
    }

    static void TestSub()
    {
        AssertEqual(
            M(new float[,] { { 5, 5 } }) - M(new float[,] { { 2, 3 } }),
            M(new float[,] { { 3, 2 } }));
    }

    static void TestMulElementwise()
    {
        AssertEqual(
            M(new float[,] { { 2, 3 } }) * M(new float[,] { { 4, 5 } }),
            M(new float[,] { { 8, 15 } }));
    }

    static void TestDivElementwise()
    {
        AssertEqual(
            M(new float[,] { { 10, 20 } }) / M(new float[,] { { 2, 4 } }),
            M(new float[,] { { 5, 5 } }));
    }

    static void TestAddNegativeCancel()
    {
        Matrix a = M(new float[,] { { -5, -3 } });
        Matrix b = M(new float[,] { { 5, 3 } });
        AssertEqual(a + b, M(new float[,] { { 0, 0 } }));
    }

    static void TestSubResultNegative()
    {
        AssertEqual(
            M(new float[,] { { 1, 2 } }) - M(new float[,] { { 3, 5 } }),
            M(new float[,] { { -2, -3 } }));
    }

    static void TestMulByZero()
    {
        Matrix a = M(new float[,] { { 99, -99 } });
        Matrix z = M(new float[,] { { 0, 0 } });
        AssertEqual(a * z, z);
    }

    static void TestDivByOne()
    {
        Matrix a = M(new float[,] { { 7, -3 } });
        AssertEqual(a / M(new float[,] { { 1, 1 } }), a);
    }

    static void TestAddSelf()
    {
        // A + A == 2*A  (broadcast scalar 2)
        Matrix a = M(new float[,] { { 3, 7 }, { -1, 5 } });
        Matrix e = M(new float[,] { { 6, 14 }, { -2, 10 } });
        AssertEqual(a + a, e);
    }

    static void TestSubSelf()
    {
        Matrix a = M(new float[,] { { 3, 7 }, { -1, 5 } });
        Matrix z = M(new float[,] { { 0, 0 }, { 0, 0 } });
        AssertEqual(a - a, z);
    }

    static void TestMulSelf()
    {
        Matrix a = M(new float[,] { { 2, 3 } });
        AssertEqual(a * a, M(new float[,] { { 4, 9 } }));
    }

    static void TestDivSelf()
    {
        Matrix a = M(new float[,] { { 5, -7 } });
        AssertEqual(a / a, M(new float[,] { { 1, 1 } }));
    }

    // =========================================================================
    //  BROADCAST
    // =========================================================================

    static void TestBroadcastScalarPlusRow()
    {
        AssertEqual(
            Scalar(1) + M(new float[,] { { 10, 20, 30 } }),
            M(new float[,] { { 11, 21, 31 } }));
    }

    static void TestBroadcastScalarMulMatrix()
    {
        AssertEqual(
            Scalar(3) * M(new float[,] { { 1, 2 }, { 3, 4 } }),
            M(new float[,] { { 3, 6 }, { 9, 12 } }));
    }

    static void TestBroadcastColPlusRow()
    {
        Matrix col = M(new float[,] { { 10 }, { 20 }, { 30 } }); // 3×1
        Matrix row = M(new float[,] { { 1, 2, 3 } });             // 1×3
        AssertEqual(col + row, M(new float[,] {
            { 11, 12, 13 },
            { 21, 22, 23 },
            { 31, 32, 33 }
        }));
    }

    static void TestBroadcastColMulRow()
    {
        Matrix col = M(new float[,] { { 2 }, { 3 } });   // 2×1
        Matrix row = M(new float[,] { { 4, 5 } });        // 1×2
        AssertEqual(col * row, M(new float[,] { { 8, 10 }, { 12, 15 } }));
    }

    static void TestBroadcastOneByOneRow()
    {
        AssertEqual(
            Scalar(5) + M(new float[,] { { 1, 2, 3 } }),
            M(new float[,] { { 6, 7, 8 } }));
    }

    static void TestBroadcastColPlusScalar()
    {
        Matrix col = M(new float[,] { { 1 }, { 2 }, { 3 } });
        AssertEqual(
            col + Scalar(10),
            M(new float[,] { { 11 }, { 12 }, { 13 } }));
    }

    // =========================================================================
    //  EXTEND
    // =========================================================================

    static void TestExtendColumn()
    {
        Matrix a = M(new float[,] { { 1 }, { 2 } });
        Matrix r = new Extend(FunctionType.Column).Apply((a, 3));
        AssertEqual(r, M(new float[,] { { 1, 1, 1 }, { 2, 2, 2 } }));
    }

    static void TestExtendRow()
    {
        Matrix a = M(new float[,] { { 1, 2 } });
        Matrix r = new Extend(FunctionType.Row).Apply((a, 3));
        AssertEqual(r, M(new float[,] { { 1, 2 }, { 1, 2 }, { 1, 2 } }));
    }

    static void TestExtendColumnByOne()
    {
        Matrix a = M(new float[,] { { 7 }, { 8 } });
        Matrix r = new Extend(FunctionType.Column).Apply((a, 1));
        AssertEqual(r, a);
    }

    static void TestExtendRowByOne()
    {
        Matrix a = M(new float[,] { { 3, 4 } });
        Matrix r = new Extend(FunctionType.Row).Apply((a, 1));
        AssertEqual(r, a);
    }

    static void TestExtendScalarColumn()
    {
        Matrix a = Scalar(5);
        Matrix r = new Extend(FunctionType.Column).Apply((a, 4));
        AssertEqual(r, M(new float[,] { { 5, 5, 5, 5 } }));
    }

    static void TestExtendScalarRow()
    {
        Matrix a = Scalar(5);
        Matrix r = new Extend(FunctionType.Row).Apply((a, 4));
        AssertShape(r, 4, 1);
        for (int x = 0; x < 4; x++)
            AssertTrue(MathF.Abs(r.values[x, 0] - 5f) < 1e-4f, $"Row {x} != 5");
    }

    // =========================================================================
    //  GETVECTORS
    // =========================================================================

    static void TestGetRowsBasic()
    {
        Matrix a = M(new float[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
        Matrix r = a.GetRows(new[] { 2, 0 });
        Matrix e = M(new float[,] { { 7, 8, 9 }, { 1, 2, 3 } }).T();
        AssertEqual(r, e);
    }

    static void TestGetColumnsBasic()
    {
        Matrix a = M(new float[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
        Matrix r = a.GetColumns(new[] { 2, 0 });
        Matrix e = M(new float[,] { { 3, 1 }, { 6, 4 }, { 9, 7 } }).T();
        AssertEqual(r, e);
    }

    static void TestGetRowsDuplicates()
    {
        Matrix a = M(new float[,] { { 1, 2 }, { 3, 4 } });
        AssertEqual(a.GetRows(new[] { 1, 1 }), M(new float[,] { { 3, 4 }, { 3, 4 } }).T());
    }

    static void TestGetColumnsDuplicates()
    {
        Matrix a = M(new float[,] { { 1, 2 }, { 3, 4 } });
        AssertEqual(a.GetColumns(new[] { 1, 1 }), M(new float[,] { { 2, 2 }, { 4, 4 } }).T());
    }

    static void TestGetRowsSingle()
    {
        Matrix a = M(new float[,] { { 9, 8, 7 }, { 6, 5, 4 } });
        Matrix r = a.GetRows(new[] { 0 });
        AssertEqual(r, M(new float[,] { { 9 }, { 8 }, { 7 } }));
    }

    static void TestGetColumnsSingle()
    {
        Matrix a = M(new float[,] { { 9, 8 }, { 7, 6 }, { 5, 4 } });
        AssertEqual(a.GetColumns(new[] { 1 }), M(new float[,] { { 8 }, { 6 }, { 4 } }));
    }

    static void TestGetRowsAllInOrder()
    {
        Matrix a = M(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        AssertEqual(a.GetRows(new[] { 0, 1, 2 }), a.T());
    }

    static void TestGetColumnsAllInOrder()
    {
        Matrix a = M(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        AssertEqual(a.GetColumns(new[] { 0, 1, 2 }), a.T());
    }

    static void TestGetRowsReverse()
    {
        Matrix a = M(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        Matrix r = a.GetRows(new[] { 2, 1, 0 });
        AssertEqual(r, M(new float[,] { { 5, 6 }, { 3, 4 }, { 1, 2 } }).T());
    }

    static void TestGetColumnsReverse()
    {
        Matrix a = M(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        Matrix r = a.GetColumns(new[] { 2, 1, 0 });
        AssertEqual(r, M(new float[,] { { 3, 2, 1 }, { 6, 5, 4 } }).T());
    }

    static void TestGetRowsOnTransposed()
    {
        Matrix a = M(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        // A^T is 3×2; row 1 is [2, 5]
        Matrix r = a.T().GetRows(new[] { 1 });
        AssertEqual(r, M(new float[,] { { 2 }, { 5 } }));
    }

    static void TestGetColumnsOnTransposed()
    {
        Matrix a = M(new float[,] { { 10, 20, 30 }, { 40, 50, 60 } });
        // A^T is 3×2; column 0 is [10, 20, 30]
        Matrix r = a.T().GetColumns(new[] { 0 });
        AssertEqual(r, M(new float[,] { { 10 }, { 20 }, { 30 } }));
    }

    static void TestGetRowsAfterAdd()
    {
        Matrix a = M(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix s = a + M(new float[,] { { 1, 1 }, { 1, 1 } });
        AssertEqual(s.GetRows(new[] { 0 }), M(new float[,] { { 2 }, { 3 } }));
    }

    static void TestGetColumnsAfterMul()
    {
        Matrix a = M(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix s = a * M(new float[,] { { 2, 2 }, { 2, 2 } });
        AssertEqual(s.GetColumns(new[] { 1 }), M(new float[,] { { 4 }, { 8 } }));
    }

    // =========================================================================
    //  CONCATENATE
    // =========================================================================

    static void TestConcatHorizontal()
    {
        Matrix a = M(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix b = M(new float[,] { { 5, 6 }, { 7, 8 } });
        Matrix r = new Concatenate(FunctionType.Row).Apply(new Matrix[] { a, b });
        Matrix e = M(new float[,] { { 1, 2, 5, 6 }, { 3, 4, 7, 8 } });
        AssertEqual(r, e);
    }

    static void TestConcatVertical()
    {
        Matrix a = M(new float[,] { { 1, 2 } });
        Matrix b = M(new float[,] { { 3, 4 } });
        Matrix r = new Concatenate(FunctionType.Column).Apply(new Matrix[] { a, b });
        Matrix e = M(new float[,] { { 1, 2 }, { 3, 4 } });
        AssertEqual(r, e);
    }

    static void TestConcatSingleRow()
    {
        Matrix a = M(new float[,] { { 1, 2, 3 } });
        Matrix b = M(new float[,] { { 4, 5, 6 } });
        Matrix r = new Concatenate(FunctionType.Column).Apply(new Matrix[] { a, b });
        AssertShape(r, 2, 3);
    }

    static void TestConcatThenTranspose()
    {
        Matrix a = M(new float[,] { { 1, 2 } });
        Matrix b = M(new float[,] { { 3, 4 } });
        Matrix r = new Concatenate(FunctionType.Column).Apply(new Matrix[] { a, b }).T();
        AssertShape(r, 2, 2);
    }

    // =========================================================================
    //  BSUM
    // =========================================================================

    static void TestSum()
    {
        Matrix a = M(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        Matrix r = new SymmetricApplication<Add>().Apply(a);

        AssertTrue(MathF.Abs(r.values[0,0] - 21f) < 1e-3f, $"BSum row total {r.values[0, 0]} != 21");
    }

    // =========================================================================
    //  MATRIX MULTIPLICATION
    // =========================================================================

    static void TestMatMulSquare()
    {
        Matrix a = M(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix b = M(new float[,] { { 5, 6 }, { 7, 8 } });
        AssertEqual(a >> b, M(new float[,] { { 19, 22 }, { 43, 50 } }));
    }

    static void TestMatMulRectA()
    {
        Matrix a = M(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        Matrix b = M(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        AssertEqual(a >> b, M(new float[,] { { 22, 28 }, { 49, 64 } }));
    }

    static void TestMatMulRectB()
    {
        Matrix a = M(new float[,] { { 1 }, { 2 }, { 3 } });
        Matrix b = M(new float[,] { { 4, 5, 6 } });
        AssertEqual(a >> b, M(new float[,] { { 4, 5, 6 }, { 8, 10, 12 }, { 12, 15, 18 } }));
    }

    static void TestMatMulChain()
    {
        Matrix a = M(new float[,] { { 1, 2 } });
        Matrix b = M(new float[,] { { 3 }, { 4 } });
        AssertEqual((a >> b) >> Scalar(5), Scalar(55));
    }

    static void TestMatMulIdentityLeft()
    {
        Matrix a = M(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix I = M(new float[,] { { 1, 0 }, { 0, 1 } });
        AssertEqual(I >> a, a);
    }

    static void TestMatMulIdentityRight()
    {
        Matrix a = M(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix I = M(new float[,] { { 1, 0 }, { 0, 1 } });
        AssertEqual(a >> I, a);
    }

    static void TestMatMulZeroLeft()
    {
        Matrix a = M(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix z = M(new float[,] { { 0, 0 }, { 0, 0 } });
        AssertEqual(z >> a, z);
    }

    static void TestMatMulZeroRight()
    {
        Matrix a = M(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix z = M(new float[,] { { 0, 0 }, { 0, 0 } });
        AssertEqual(a >> z, z);
    }

    static void TestMatMulScalars()
    {
        AssertEqual(Scalar(3) >> Scalar(7), Scalar(21));
    }

    static void TestMatMulDotProduct()
    {
        Matrix row = M(new float[,] { { 1, 2, 3 } });
        Matrix col = M(new float[,] { { 4 }, { 5 }, { 6 } });
        AssertEqual(row >> col, Scalar(32));
    }

    static void TestMatMulOuterProduct()
    {
        Matrix col = M(new float[,] { { 1 }, { 2 }, { 3 } });
        Matrix row = M(new float[,] { { 4, 5 } });
        AssertEqual(col >> row, M(new float[,] { { 4, 5 }, { 8, 10 }, { 12, 15 } }));
    }

    static void TestMatMulSquare3x3()
    {
        Matrix a = M(new float[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
        Matrix I = M(new float[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } });
        AssertEqual(a >> I, a);
        AssertEqual(I >> a, a);
    }

    static void TestMatMulAssociative()
    {
        // (AB)C == A(BC)
        Matrix a = M(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix b = M(new float[,] { { 5, 6 }, { 7, 8 } });
        Matrix c = M(new float[,] { { 1, 0 }, { 0, 1 } });
        AssertEqual((a >> b) >> c, a >> (b >> c), 1e-3f);
    }

    static void TestMatMulTransposeReversal()
    {
        Matrix a = M(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }); // 3×2
        Matrix b = M(new float[,] { { 7, 8, 9 }, { 10, 11, 12 } });   // 2×3
        AssertEqual((a >> b).T(), b.T() >> a.T(), 1e-3f);
    }

    static void TestMatMulGram()
    {
        // Gram matrix A^T A must be symmetric
        Matrix a = M(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } }); // 2×3
        Matrix g = a.T() >> a; // 3×3
        AssertShape(g, 3, 3);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                if (MathF.Abs(g.values[i, j] - g.values[j, i]) > 1e-3f)
                    throw new Exception($"Gram matrix not symmetric at ({i},{j})");
    }

    // =========================================================================
    //  MODULE: Apply (wraps ISingleValueFunction)
    // =========================================================================

    static void TestApplySin()
    {
        Matrix a = M(new float[,] { { MathF.PI / 2 } });
        AssertEqual(new Apply<Sin>().Forward(a), Scalar(1), 1e-3f);
    }

    static void TestApplyCos()
    {
        AssertEqual(new Apply<Cos>().Forward(Scalar(0)), Scalar(1), 1e-3f);
    }

    static void TestApplyExp()
    {
        AssertEqual(new Apply<Exp>().Forward(Scalar(0)), Scalar(1), 1e-3f);
    }

    static void TestApplyLog()
    {
        AssertEqual(new Apply<Log>().Forward(Scalar(MathF.E)), Scalar(1), 1e-3f);
    }

    static void TestApplyAbs()
    {
        AssertEqual(new Apply<Abs>().Forward(Scalar(-7)), Scalar(7));
    }

    static void TestApplyTanh()
    {
        AssertEqual(new Apply<TanH>().Forward(Scalar(0)), Scalar(0), 1e-4f);
    }

    static void TestApplyTan()
    {
        AssertEqual(new Apply<Tan>().Forward(Scalar(0)), Scalar(0), 1e-4f);
    }

    // =========================================================================
    //  MODULE: SoftMax
    // =========================================================================

    static void TestSoftMaxSumsToOne()
    {
        Matrix a = M(new float[,] { { 1, 2, 3, 4 } });
        Matrix r = new SoftMax().Forward(a);
        float sum = 0;
        for (int x = 0; x < r.shape.width; x++)
            for (int y = 0; y < r.shape.height; y++)
                sum += r.values[x, y];
        AssertTrue(MathF.Abs(sum - 1f) < 1e-4f, $"SoftMax sum={sum}, expected 1");
    }

    static void TestSoftMaxEqualInputs()
    {
        // Equal inputs → equal outputs → each = 1/n
        Matrix a = M(new float[,] { { 1, 1, 1, 1 } });
        Matrix r = new SoftMax().Forward(a);
        float expected = 0.25f;
        for (int x = 0; x < r.shape.width; x++)
            for (int y = 0; y < r.shape.height; y++)
                AssertTrue(MathF.Abs(r.values[x, y] - expected) < 1e-4f,
                    $"SoftMax equal-input entry [{x},{y}]={r.values[x, y]}, expected {expected}");
    }

    static void TestSoftMaxRange()
    {
        Matrix a = M(new float[,] { { -10, 0, 5, 100 } });
        Matrix r = new SoftMax().Forward(a);
        for (int x = 0; x < r.shape.width; x++)
            for (int y = 0; y < r.shape.height; y++)
            {
                float v = r.values[x, y];
                AssertTrue(v >= 0f && v <= 1f, $"SoftMax output out of (0,1): {v}");
            }
    }

    static void TestSoftMaxDominant()
    {
        // Very large last element → its softmax ≈ 1
        Matrix a = M(new float[,] { { 0, 0, 100 } });
        Matrix r = new SoftMax().Forward(a);
        float last = 0;
        // Last valid column index depends on layout; find max
        for (int x = 0; x < r.shape.width; x++)
            for (int y = 0; y < r.shape.height; y++)
                last = MathF.Max(last, r.values[x, y]);
        AssertTrue(last > 0.99f, $"SoftMax dominant value {last} not ≈ 1");
    }

    // =========================================================================
    //  MODULE: Activation
    // =========================================================================

    static void TestSigmoidAtZero()
    {
        Matrix r = new Sigmoid().Forward(Scalar(0));
        AssertTrue(MathF.Abs(r.values[0, 0] - 0.5f) < 1e-4f, $"Sigmoid(0)={r.values[0, 0]}, expected 0.5");
    }

    static void TestSigmoidLarge()
    {
        Matrix r = new Sigmoid().Forward(Scalar(100f));
        AssertTrue(r.values[0, 0] > 0.99f, $"Sigmoid(100)={r.values[0, 0]}, expected ≈1");
    }

    static void TestSigmoidNegative()
    {
        Matrix r = new Sigmoid().Forward(Scalar(-100f));
        AssertTrue(r.values[0, 0] < 0.01f, $"Sigmoid(-100)={r.values[0, 0]}, expected ≈0");
    }

    static void TestSwishAtZero()
    {
        // Swish(x) = x * sigmoid(x), so Swish(0) = 0
        Matrix r = new Swish().Forward(Scalar(0));
        AssertTrue(MathF.Abs(r.values[0, 0]) < 1e-4f, $"Swish(0)={r.values[0, 0]}, expected 0");
    }

    // =========================================================================
    //  STABILITY / EDGE CASES
    // =========================================================================

    static void TestAllZeros()
    {
        Matrix z = M(new float[,] { { 0, 0 }, { 0, 0 } });
        AssertEqual(new Apply<Sin>().Forward(z), z);
        AssertEqual(z + z, z);
        AssertEqual(z >> z, z);
    }

    static void TestAllOnesMatMul()
    {
        Matrix a = M(new float[,] { { 1, 1 }, { 1, 1 } });
        AssertEqual(a >> a, M(new float[,] { { 2, 2 }, { 2, 2 } }));
    }

    static void TestAllNegOnesMatMul()
    {
        // (-1)*(-1) + (-1)*(-1) = 2 per entry
        Matrix a = M(new float[,] { { -1, -1 }, { -1, -1 } });
        AssertEqual(a >> a, M(new float[,] { { 2, 2 }, { 2, 2 } }));
    }

    static void TestScalarAllOps()
    {
        Matrix a = Scalar(5);
        Matrix b = Scalar(3);
        AssertEqual(a + b, Scalar(8));
        AssertEqual(a - b, Scalar(2));
        AssertEqual(a * b, Scalar(15));
        AssertEqual(a / b, Scalar(5f / 3f), 1e-4f);
        AssertEqual(a >> b, Scalar(15));
        AssertEqual(a.T(), Scalar(5));
    }

    static void TestLargeValuesAbs()
    {
        Matrix a = M(new float[,] { { 1e6f, -1e6f } });
        AssertEqual(new Apply<Abs>().Forward(a), M(new float[,] { { 1e6f, 1e6f } }));
    }

    static void TestLargeMatMul()
    {
        // 4×4 identity check at slightly larger scale
        Matrix a = M(new float[,] {
            { 1, 2, 3, 4 },
            { 5, 6, 7, 8 },
            { 9, 10, 11, 12 },
            { 13, 14, 15, 16 }
        });
        Matrix I = M(new float[,] {
            { 1, 0, 0, 0 },
            { 0, 1, 0, 0 },
            { 0, 0, 1, 0 },
            { 0, 0, 0, 1 }
        });
        AssertEqual(a >> I, a, 1e-3f);
        AssertEqual(I >> a, a, 1e-3f);
    }
}
