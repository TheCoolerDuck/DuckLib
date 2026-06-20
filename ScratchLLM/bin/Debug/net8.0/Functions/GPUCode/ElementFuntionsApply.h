
__device__ float apply(float x, int ID)
{
    switch(ID)
    {
        case -1932355109: return x < 0 ? -x : x;
        case 1932355109: return x < 0 ? -1 : 1;;
        case 1578662460: return cosf(x);
        case -1578662460: return -sinf(x);
        case -676624440: return expf(x);
        case 676624440: return expf(x);
        case -1579320335: return logf(x);
        case 1579320335: return 1.0f / x;
        case 1119638765: return sinf(x);
        case -1119638765: return cosf(x);
        case -8167624: return tanf(x);
        case 8167624: return 1.0f / (cosf(x) * cosf(x));
        case -1412481712: return tanhf(x);
        case 1412481712: return 1.0f - tanhf(x) * tanhf(x);
        default: return 1.0f / 0.0f;
    }
}
