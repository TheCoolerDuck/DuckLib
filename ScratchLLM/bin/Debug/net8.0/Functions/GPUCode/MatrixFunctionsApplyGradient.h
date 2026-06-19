
__device__ float2 apply(float x, float y, int ID)
{
    switch(ID)
    {
        case -1648121324: return make_float2(1.0f, 1.0f);
        case 1248411368: return make_float2(1.0f / y, -x / (y * y));
        case 1017635769: return x > y ? make_float2(1.0f, 0.0f) : make_float2(0.0f, 1.0f);
        case 781469175: return x < y ? make_float2(1.0f, 0.0f) : make_float2(0.0f, 1.0f);
        case 1351319841: return make_float2(y, x);
        case -1120375947: return 
            {
                float p = powf(x, y);
                return make_float2(y * p / x, p * logf(x));
            };
        case 1054396597: return make_float2(1.0f, -1.0f);
        default: return make_float2(1.0f / 0.0f, 1.0f / 0.0f);
    }
}
