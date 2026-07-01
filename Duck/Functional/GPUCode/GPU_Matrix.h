struct Matrix
{
	float* values;
	int width;
	int height;
	bool transposed;

	__device__ float Get(int x, int y)
	{
		return values[Index(x, y)];
	}

	__device__ void Set(int x, int y, float value)
	{
		values[Index(x, y)] = value;
	}

	__device__ void Add(int x, int y, float value)
	{
		values[Index(x, y)] += value;
	}

	__device__ int Index(int x, int y)
	{
		if (transposed)
			return x * height + y;
		else
			return y * width + x;
	}

	__device__ int2 Quards(int i)
	{
		if (transposed)
			return make_int2(i / width, i % width);
		else
			return make_int2(i % width, i / width);
	}
};