#include <iostream>
#include <cuda.h>
using namespace std;

# define DELLEXPORT extern "C" __declspec(dllexport)
typedef double SIGNAL;
#define huge 1e32

__device__
SIGNAL _min(SIGNAL a, SIGNAL b)
{
	if (a < b)
		return a;
	else
		return b;
}
__device__
SIGNAL _max(SIGNAL a, SIGNAL b)
{
	if (a > b)
		return a;
	else
		return b;
}
__device__
SIGNAL _min(SIGNAL a, SIGNAL b, SIGNAL c)
{
	return min(min(a, b), c);
}
__device__
SIGNAL _abs(SIGNAL a)
{
	if (a < 0)
		return -a;
	else
		return a;
}
__device__
SIGNAL _dist(SIGNAL a, SIGNAL b)
{
	return _abs(a - b);
}
__device__
//Matrix index to array index
int _mitai(int i, int j, int size)
{
	return i * size + j;
}
__device__
//Matrix index to buffer index
int _mitbi(int mi, int start_index)
{
	return mi - start_index;
}
__device__
SIGNAL _save_get_value(SIGNAL* buffer, int buffer_size, int index) {
	if (index < 0 || index >= buffer_size)
		return huge;
	else
		return buffer[index];
}

__device__
SIGNAL dtw_distance(SIGNAL* a, int a_i,
	SIGNAL* b, int b_i,
	int ts_size, int window)
{
	int buffers_size = 2 * window;
	SIGNAL* first = (SIGNAL*)malloc(buffers_size * sizeof(SIGNAL));
	SIGNAL* second = (SIGNAL*)malloc(buffers_size * sizeof(SIGNAL));
	memset(first, huge, buffers_size * sizeof(SIGNAL));
	memset(second, huge, buffers_size * sizeof(SIGNAL));
	int first_start_index = -window - 1;
	int second_start_index = -window;;
	for (int i = 0; i < ts_size; i++)
	{
		SIGNAL a_value = a[_mitai(a_i, i, ts_size)];
		for (int j = second_start_index; j < _min(ts_size, i + window); j++)
		{
			SIGNAL b_value = b[_mitai(b_i, j, ts_size)];
			SIGNAL dist = _dist(a_value, b_value);
			if (i == 0 && j == 0) {
				second[window] = dist;
			}
			else if (i == 0)
			{
				int second_index = window + j;
				second[second_index] = second[second_index - 1] + dist;
			}
			else if (j == 0)
			{
				second[window - i] = first[window - i + 1] + dist;
			}
			else {
				int first_index = _mitbi(j, first_start_index);
				int second_index = _mitbi(j, second_start_index);
				second[second_index] = _min(
					_save_get_value(first, buffers_size, first_index - 1),
					_save_get_value(second, buffers_size, second_index - 1),
					_save_get_value(first, buffers_size, first_index))
					+ dist;
			}
		}
		SIGNAL *tmp = first;
		first = second;
		second = tmp;

		first_start_index++;
		second_start_index++;
	}

	SIGNAL result = first[window];
	free(first);
	free(second);
	return result;
}

__global__
void cuda_dtw(SIGNAL* a, SIGNAL* b, SIGNAL* result,
	int a_size, int b_size, int ts_size, int window)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int size = a_size * b_size;
	for (int i = index; i < size; i += stride) {
		//int start = i * ts_size;
		int a_i = i / a_size;
		int b_i = i % a_size;
		result[i] = dtw_distance(a, a_i,
			b, b_i,
			ts_size, window);
	}
}


DELLEXPORT void dtw_gpu(SIGNAL* a, SIGNAL* b, SIGNAL* result,
	int a_size, int b_size, int ts_size, int maxWindow) {

	const size_t malloc_limit = 3ull * 1024ull * 1024ull * 1024ull;
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, malloc_limit);

	SIGNAL* gpu_a;
	SIGNAL* gpu_b;
	SIGNAL* gpu_result;
	cudaMallocManaged(&gpu_a, a_size * ts_size * sizeof(SIGNAL));
	cudaMallocManaged(&gpu_b, b_size * ts_size * sizeof(SIGNAL));
	cudaMallocManaged(&gpu_result, a_size * b_size * sizeof(SIGNAL));

	cudaMemcpy(gpu_a, a, a_size * ts_size * sizeof(SIGNAL), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_b, b, b_size * ts_size * sizeof(SIGNAL), cudaMemcpyHostToDevice);

	int blockSize = 256;
	int numBlocks = ceil(((a_size * b_size) + blockSize - 1) / (1.0 * blockSize));

	cuda_dtw
		<< <numBlocks, blockSize >> >
		(gpu_a, gpu_b, gpu_result,
			a_size, b_size, ts_size, maxWindow
			);
	cudaDeviceSynchronize();
	cudaMemcpy(result, gpu_result, a_size * b_size * sizeof(SIGNAL), cudaMemcpyDeviceToHost);

	cudaFree(gpu_a);
	cudaFree(gpu_b);
	cudaFree(gpu_result);
}