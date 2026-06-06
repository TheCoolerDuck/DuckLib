struct Matrix
{
	float* values;
	int width;
	int height;
	bool transposed;

	__device__ float Get(int x, int y)
	{
		return values[index(x, y)];
	}

	__device__ void Set(int x, int y, float value)
	{
		printf("set x:%d y:%d v:%f \\\n", x, y, value);
		values[index(x, y)] = value;
	}

	__device__ void Add(int x, int y, float value)
	{
		values[y * height + x] += value;
	}

	__device__ int index(int x, int y)
	{
		if (transposed)
			return y * height + x;
		else
			return x * width + y;
	}
};