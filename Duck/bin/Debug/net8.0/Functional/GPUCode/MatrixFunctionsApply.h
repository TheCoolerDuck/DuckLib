
__device__ float apply(float x, float y, int ID)
{
    switch(ID)
    {
        case -1648121324: return x + y;
        case 1248411368: return x / y;
        case 1017635769: return x > y ? x : y;
        case 781469175: return x < y ? x : y;
        case 1351319841: return x * y;
        case -1120375947: return powf(x, y);
        case 1054396597: return x - y;
        case 1151562531: return x % y;
        default: return 1.0f / 0.0f;
    }
}
