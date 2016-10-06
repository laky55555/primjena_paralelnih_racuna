#include <stdio.h>
#include "cuda_wrapper.h"


#define BLOCK_SIZE 32


__device__	double	result2 = 0.0;


__device__	double	atomicMax(double *address, double val)
{
	unsigned long long int *address_to_ull = (unsigned long long *) address;
	unsigned long long int	old;
	unsigned long long int	compare;

	if (val == 0.0)
		return *address;

	do {
		compare = *address_to_ull;
		old = atomicCAS(address_to_ull, compare, __double_as_longlong(fmax(val , __longlong_as_double(compare))));
	} while (old != compare);

	return __longlong_as_double(old);
}
		

template <unsigned int velicina_bloka>
__global__ void reduction_jedan_blok(double *in_data, int N)
{

	__shared__ double smem2[velicina_bloka];

	unsigned int indeks_u_redku = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int prvi_indeks_reda = gridDim.x * blockDim.x * blockIdx.y;
	unsigned int ukupni_indeks = indeks_u_redku + prvi_indeks_reda;
	
	smem2[threadIdx.x] = 0;
	if(indeks_u_redku < N)
	{
		double dijagonala = -1;

		smem2[threadIdx.x] = fabs(in_data[ukupni_indeks]);
		if(indeks_u_redku == blockIdx.y)
		{
			dijagonala = smem2[threadIdx.x];
			smem2[threadIdx.x] = 0;
		}
		__syncthreads();


		if (velicina_bloka >= 1024 && threadIdx.x < 512)
			smem2[threadIdx.x] = fmax(smem2[threadIdx.x], smem2[threadIdx.x + 512]);
		__syncthreads();

		if (velicina_bloka >=  512 && threadIdx.x < 256)
			smem2[threadIdx.x] = fmax(smem2[threadIdx.x], smem2[threadIdx.x + 256]);
		__syncthreads();

		if (velicina_bloka >=  256 && threadIdx.x < 128)
			smem2[threadIdx.x] = fmax(smem2[threadIdx.x], smem2[threadIdx.x + 128]);
		__syncthreads();

		if (velicina_bloka >=  128 && threadIdx.x < 64)
			smem2[threadIdx.x] = fmax(smem2[threadIdx.x], smem2[threadIdx.x + 64]);
		__syncthreads();

			
		if (threadIdx.x < 32) {
			volatile double *tmp = smem2;

			tmp[threadIdx.x] = fmax(tmp[threadIdx.x], tmp[threadIdx.x + 32]);
			tmp[threadIdx.x] = fmax(tmp[threadIdx.x], tmp[threadIdx.x + 16]);
			tmp[threadIdx.x] = fmax(tmp[threadIdx.x], tmp[threadIdx.x + 8]);
			tmp[threadIdx.x] = fmax(tmp[threadIdx.x], tmp[threadIdx.x + 4]);
			tmp[threadIdx.x] = fmax(tmp[threadIdx.x], tmp[threadIdx.x + 2]);
			tmp[threadIdx.x] = fmax(tmp[threadIdx.x], tmp[threadIdx.x + 1]);
		}
		
		if (threadIdx.x == 0)
		{	
			if(dijagonala == -1)
				dijagonala = fabs(in_data[prvi_indeks_reda + blockIdx.y]);
			smem2[0] += dijagonala;
			atomicMax(&result2, smem2[0]);
		}
	}
}


void ispis_matrice (double *a, int x, int y)
{
    int i, j;
    for(i=0; i<x; ++i)
    {
    	for(j=0; j<y; ++j)
		printf("%lg ", a[j*x+i]);
        printf("\n");
    }
    printf("\n");
}

void izracunaj_normalno(double *a, int n)
{
    double max=-1;
    for(int i=0; i<n; i++)
        for(int j=0; j<n; j++)
            if(i!=j && a[i+n*j] + a[i*n+i] > max)
                max = a[i+n*j] + a[i*n+i];

    printf("Normalno racunanje daje rezultat: %lg.\n", max);
}

void jednake(double *a, double *b, int size)
{
    for(int i=0; i<size; i++)
        if(a[i] != b[i])
        {
            printf("Mjesto greske je %d, a brojevi su: %lg i %lg\n", i, a[i], b[i]);
            return;
	}
    
	printf("Matrice su jednake.\n");
}

__global__ void cuda_transp (double *a, double *a_trans, const int N, const int lda)
{ 
	__shared__ double smem[BLOCK_SIZE][BLOCK_SIZE+1];

	unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(ix < N && iy < N)
	{
		smem[threadIdx.x][threadIdx.y] = a[iy * lda + ix];
    	a_trans[iy + lda * ix] = smem[threadIdx.x][threadIdx.y];
	}
 
}


void transponiraj(double *a, double *b, int N)
{
	for(int i=0; i<N; i++)
		for(int j=0; j<N; j++)
			b[i+N*j] = a[i*N+j];


}


int main(int argc, char **argv)
{

    dim3 grid_size, block_size;
	int N, size;
	double *hst_A, *hst_A_transp;
	double *dev_A, *dev_A_transp;
	double norma2;
	FILE *fA;

	size_t	pitch;
	int		lda;

    double gpu_prebacivanje_podataka = 0.0, gpu_trans = 0.0, gpu_redukcija_jedan_blok = 0.0, cpu = 0.0;


	if(argc != 3)
	{
		fprintf(stderr, "Za pokretanje %s program ocekuje 2 inta za dimenzije matrica, te 3 stringa koji oznacavaju imena datoteka iz kojih citamo, pisemo\n", argv[0]);
		return 0;
	}


	N = atoi(argv[1]);
	size = N*N;

    open_file(fA, argv[2], "r");

    // alokacija memorije na hostu
    host_alloc(hst_A, double, size * sizeof(double));
    host_alloc(hst_A_transp, double, size * sizeof(double));
    
    // alokacija memorije na deviceu
    cuda_exec(cudaMallocPitch(&dev_A, &pitch, N * sizeof(double), N));
    cuda_exec(cudaMallocPitch(&dev_A_transp, &pitch, N * sizeof(double), N));
    lda = pitch / sizeof(double);

    // citanje podataka iz binarne datoteke
    read_file(hst_A, sizeof(double), size, fA);
   

    cpu -= timer();
    izracunaj_normalno(hst_A, N);
    cpu += timer();


    // kopiranje podataka na device
    gpu_prebacivanje_podataka -= timer();
    cuda_exec(cudaMemcpy2D(dev_A, pitch, hst_A, N * sizeof(double), N * sizeof(double), N, cudaMemcpyHostToDevice));
    gpu_prebacivanje_podataka += timer();


	block_size = dim3(32, 32, 1);
	grid_size.x = min((lda + block_size.x - 1) / block_size.x, 65535);
	grid_size.y = min((N + block_size.y - 1) / block_size.y, 65535);
	printf("Dimenzije su: grid (%d, %d), blok (%d, %d), lda = %d.\n", grid_size.x, grid_size.y, block_size.x, block_size.y, lda);

    gpu_trans -= timer();
    cuda_transp<<<grid_size, block_size>>>(dev_A, dev_A_transp, N, lda);
    cuda_exec(cudaDeviceSynchronize());
    gpu_trans += timer();


    //provjera transponiranja
    //transponira matricu A i spremi je u A_transp
    transponiraj(hst_A, hst_A_transp, N);
    cuda_exec(cudaMemcpy2D(hst_A, N * sizeof(double), dev_A_transp, pitch, N * sizeof(double), N, cudaMemcpyDeviceToHost));
    jednake(hst_A, hst_A_transp, N);
    printf("GPU transponiranje time: %#.4lgms\n", gpu_trans*1000);



    block_size.x = 64;
    while( (lda % (block_size.x*2)) == 0 && block_size.x <1024)
    	block_size.x *= 2;

	block_size.y = 1;

	grid_size.x = min((lda + block_size.x - 1) / block_size.x, 65535);
	grid_size.y = N;

	printf("Dimenzije su: grid (%d, %d), blok (%d, %d), lda = %d.\n", grid_size.x, grid_size.y, block_size.x, block_size.y, lda);
 
    
    gpu_redukcija_jedan_blok -= timer();
    switch (block_size.x) 
    {
		case 1024:
			reduction_jedan_blok<1024><<<grid_size, block_size>>>(dev_A_transp, N);
			break;
		case  512:
			reduction_jedan_blok< 512><<<grid_size, block_size>>>(dev_A_transp, N);
			break;
		case  256: 
			reduction_jedan_blok< 256><<<grid_size, block_size>>>(dev_A_transp, N);
			break;
		case  128: 
			reduction_jedan_blok< 128><<<grid_size, block_size>>>(dev_A_transp, N);
			break;
		case   64: 
			reduction_jedan_blok<  64><<<grid_size, block_size>>>(dev_A_transp, N);
			break;

	}

    gpu_redukcija_jedan_blok += timer();

    cuda_exec(cudaMemcpyFromSymbol(&norma2, result2, sizeof(double)));

    
    
    printf("GPU redukcija jedan blok: %#.4lgms\n", gpu_redukcija_jedan_blok*1000);
    printf("Norma pomocu GPU je: %lg\n", norma2);


    close_file(fA);
    host_free(hst_A); host_free(hst_A_transp);
    cuda_exec(cudaFree(dev_A)); cuda_exec(cudaFree(dev_A_transp));
    

	return 0;
}

