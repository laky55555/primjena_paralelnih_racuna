#include  "cuda_wrapper.h"
#include <stdio.h>
#include <float.h>


#define BLOCK_SIZE 512



__global__ void prefix(double *a, double *b, const int dim)
{
	//__shared__ double temp[BLOCK_SIZE];
	__shared__ double temp[BLOCK_SIZE*2];

	int offset = 1;
	temp[2*threadIdx.x] = a[2*threadIdx.x];
	temp[2*threadIdx.x+1] = a[2*threadIdx.x+1];
	//temp[threadIdx.x] = a[threadIdx.x + blockDim.x * blockIdx.x];


	for (int d = dim >> 1; d > 0; d >>= 1)
	{
		__syncthreads();

		if (threadIdx.x < d)
		{
			int index_a = offset * (2 * threadIdx.x + 1) - 1;
			int index_b = offset * (2 * threadIdx.x + 2) - 1;

			temp[index_b] += temp[index_a];

		}

		offset <<= 1;
	}

	if(threadIdx.x == 0)
		temp[dim - 1] = 0;

	for (int d = 1; d < dim; d <<= 1)
	{

		offset >>= 1;
		__syncthreads();

		if (threadIdx.x < d)
		{
			int index_a = offset * (2 * threadIdx.x + 1) - 1;
			int index_b = offset * (2 * threadIdx.x + 2) - 1;

			double t = temp[index_a];
			temp[index_a] = temp[index_b];
			temp[index_b] += t;
		}

	}

	__syncthreads();

	b[2*threadIdx.x] = temp[2*threadIdx.x];
	b[2*threadIdx.x+1] = temp[2*threadIdx.x+1];
	//b[threadIdx.x + blockDim.x * blockIdx.x] = temp[threadIdx.x];

}


void print_vector(int dim, double *a)
{
	for(int i = 0; i < dim; i++)
		printf("%lg ", a[i]);
	printf("\n");
}

void host_scan(int dim, double *a, double *b)
{
	b[0] = 0;
	for(int i = 1; i < dim; i++)
		b[i] = a[i-1] + b[i-1];
}

void check_vectors(int dim, double *a, double *b)
{
	for (size_t i = 0; i < dim; i++) {
		if (abs(a[i] - b[i]) >= 100*DBL_EPSILON)
			printf("On place %d, a[%d] = %lg, b[%d] = %lg\n", i, i, a[i], i, b[i]);

	}
}





int main(int argc, char const *argv[]) {

	if (argc != 3) {
		fprintf(stderr, "Program %s expects positive integer for dimension, and vector of real numbers dimension from first argument.\n", argv[0]);
		return -1;
	}

		cuda_exec(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

	dim3 grid_size, block_size;

	FILE *fA;

	double *hst_d;
	double *hst_rj;
	double *hst_rj_dev_copy;
	double *hst_temp_dev_copy;

	double *dev_d;
	//double *dev_temp;
	double *dev_rj;

	int dimension = atoi(argv[1]);

	host_alloc(hst_d, double, dimension * sizeof(double));
	host_alloc(hst_rj_dev_copy, double, dimension * sizeof(double));
	host_alloc(hst_rj, double, dimension * sizeof(double));
	host_alloc(hst_temp_dev_copy, double, dimension * sizeof(double));

	cuda_exec(cudaMalloc(&dev_d, dimension * sizeof(double)));
	//cuda_exec(cudaMalloc(&dev_temp, dimension * sizeof(double)));
	cuda_exec(cudaMalloc(&dev_rj, dimension * sizeof(double)));

	block_size = dim3(BLOCK_SIZE, 1, 1);
	grid_size.x = (dimension + block_size.x - 1) / block_size.x;
	printf("Dimensions are: grid (%d, 0, 0), blok (%d, 0, 0), matrix is %dx%d\n", grid_size.x, block_size.x, dimension, dimension);
	if(grid_size.x > 1024)
	{
		fprintf(stderr, "grid_size.x = %d, but must be under 1024.\n", grid_size.x);
		goto end;
	}

	open_file(fA, argv[2], "r");

	read_file(hst_d, sizeof(double), dimension, fA);

	cuda_exec(cudaMemcpy(dev_d, hst_d, dimension * sizeof(double), cudaMemcpyHostToDevice));


	prefix<<<grid_size, block_size>>>(dev_d, dev_rj, dimension);
	cudaDeviceSynchronize();
	cuda_exec(cudaMemcpy(hst_rj_dev_copy, dev_rj, dimension * sizeof(double), cudaMemcpyDeviceToHost));

	host_scan(dimension, hst_d, hst_rj);

	printf("Pocetni vektor\n");
	print_vector(dimension, hst_d);
	printf("Scan host vector\n");
	print_vector(dimension, hst_rj);
	printf("Scan dev vektor\n");
	print_vector(dimension, hst_rj_dev_copy);
	//printf("Provjera vektora prvog bloka\n");
	//check_vectors(dimension, hst_rj, hst_temp_dev_copy);
	printf("Provjera vektora\n");
	check_vectors(dimension, hst_rj, hst_rj_dev_copy);


end:

	free(hst_d);
	free(hst_rj);
	free(hst_rj_dev_copy);
	free(hst_temp_dev_copy);

	cuda_exec(cudaFree(dev_d));
	//cuda_exec(cudaFree(dev_temp));
	cuda_exec(cudaFree(dev_rj));


	return 0;
}
