#include <stdio.h>
#include "cuda_wrapper.h"
#include <float.h>

__constant__ double  dijagonala[8192];

__device__	double	result1 = 0.0;
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
		

template <unsigned int block_size>
__global__ void reduction_8_blokova(double *in_data, int N)
{
	double	*data = in_data + 8 * blockIdx.x * blockDim.x;

	__shared__ double smem[block_size];

	
	smem[threadIdx.x] = fmax(data[threadIdx.x + 0 * blockDim.x], data[threadIdx.x + 1 * blockDim.x]);
	smem[threadIdx.x] = fmax(smem[threadIdx.x], data[threadIdx.x + 2 * blockDim.x]);
	smem[threadIdx.x] = fmax(smem[threadIdx.x], data[threadIdx.x + 3 * blockDim.x]);
	smem[threadIdx.x] = fmax(smem[threadIdx.x], data[threadIdx.x + 4 * blockDim.x]);
	smem[threadIdx.x] = fmax(smem[threadIdx.x], data[threadIdx.x + 5 * blockDim.x]);
	smem[threadIdx.x] = fmax(smem[threadIdx.x], data[threadIdx.x + 6 * blockDim.x]);
	smem[threadIdx.x] = fmax(smem[threadIdx.x], data[threadIdx.x + 7 * blockDim.x]);
	__syncthreads();


	if (block_size >= 1024 && threadIdx.x < 512)
		smem[threadIdx.x] = fmax(smem[threadIdx.x], smem[threadIdx.x + 512]);
	__syncthreads();

	if (block_size >=  512 && threadIdx.x < 256)
		smem[threadIdx.x] = fmax(smem[threadIdx.x], smem[threadIdx.x + 256]);
	__syncthreads();

	if (block_size >=  256 && threadIdx.x < 128)
		smem[threadIdx.x] = fmax(smem[threadIdx.x], smem[threadIdx.x + 128]);
	__syncthreads();

	if (block_size >=  128 && threadIdx.x < 64)
		smem[threadIdx.x] = fmax(smem[threadIdx.x], smem[threadIdx.x + 64]);
	__syncthreads();

		
	if (threadIdx.x < 32) {
		volatile double *tmp = smem;

		tmp[threadIdx.x] = fmax(tmp[threadIdx.x], tmp[threadIdx.x + 32]);
		tmp[threadIdx.x] = fmax(tmp[threadIdx.x], tmp[threadIdx.x + 16]);
		tmp[threadIdx.x] = fmax(tmp[threadIdx.x], tmp[threadIdx.x + 8]);
		tmp[threadIdx.x] = fmax(tmp[threadIdx.x], tmp[threadIdx.x + 4]);
		tmp[threadIdx.x] = fmax(tmp[threadIdx.x], tmp[threadIdx.x + 2]);
		tmp[threadIdx.x] = fmax(tmp[threadIdx.x], tmp[threadIdx.x + 1]);
	}
	
	if (threadIdx.x == 0)
		atomicAdd(&result1, smem[0]);

}

template <unsigned int block_size>
__global__ void reduction_jedan_blok(double *in_data, int N)
{

	__shared__ double smem2[block_size];

	int broj_bloka = blockIdx.x + gridDim.x * blockIdx.y;
	smem2[threadIdx.x] = in_data[threadIdx.x + broj_bloka*block_size];
	__syncthreads();


	if (block_size >= 1024 && threadIdx.x < 512)
		smem2[threadIdx.x] = fmax(smem2[threadIdx.x], smem2[threadIdx.x + 512]);
	__syncthreads();

	if (block_size >=  512 && threadIdx.x < 256)
		smem2[threadIdx.x] = fmax(smem2[threadIdx.x], smem2[threadIdx.x + 256]);
	__syncthreads();

	if (block_size >=  256 && threadIdx.x < 128)
		smem2[threadIdx.x] = fmax(smem2[threadIdx.x], smem2[threadIdx.x + 128]);
	__syncthreads();

	if (block_size >=  128 && threadIdx.x < 64)
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
	    atomicMax(&result2, smem2[0]);

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

int jednake(double *a, double *b, int n)
{
    for(int i=0; i<n; i++)
        if(a[i] != b[i])
        {
            printf("Mjesto greske je %d, a brojevi su: %lg i %lg\n", i, a[i], b[i]);
            return 0;
	}
    return 1;
}

__global__ void zbrajanje (double *a, int N)
{

    //__shared__ double dijagonala[N];

    //gridDim.x je broj blokova u redku
    //gridDim.y je broj blokova u stupcu

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int index = blockId * blockDim.x + threadIdx.x;
	int broj_redka = threadIdx.x+ blockIdx.x * blockDim.x;

    ///Inicijalizacija shared memorije s elementima s dijagonale
    /*if(index < N)
        dijagonala[N] = a[index*N + index];
    __syncthreads();*/

    //gledamo samo elemente matrice do N (jer je matrica prosirena s nulama) i razlicite od dijagonale 
    //te elemente uvecamo za vrijednost na dijagonali
    if(broj_redka < N && blockIdx.y != broj_redka)
    {
    	//printf("Indeks je %d, indeks dijagonale %d, a=%lg, threadIdx.x=%d, dijagonala=%lg\n", index, blockIdx.y, a[index], threadIdx.x, dijagonala[blockIdx.y]);
    	a[index] = fabs(a[index]) + fabs(dijagonala[blockIdx.y]);
	}
	
	__syncthreads();

}

int nadi_odgovarajuci_broj_dretvi(int N)
{	
   // if(N <= 8)
	//return 8;
    if(N <= 64)
        return 64;
    else if(N <= 128)
        return 128;
    else if(N <= 256)
        return 256;
    else if(N <= 512)
        return 512;
    else if(N <= 768)
        return 256;
    else if(N <= 1024)
        return 1024;
    else
        return 256;
}

void prosiri_matricu(double *a, double *b, int m, int n)
{
	for(int i=0; i<m; i++)
		for(int j=0; j<n; j++)
		{
			if(j<m)
				b[i*n+j] = a[i*m+j];
			else
				b[i*n+j] = 0;
		}
}

int main(int argc, char **argv)
{

    dim3 gridDim, blockDim;
	int N, size, broj_blokova_po_redku, broj_dretvi_po_bloku, prosireni_size;
	double *hst_A, *hst_A_prosirena, *hst_dijagonala;
	double *dev_A, norma, norma2;
	FILE *fA;

    double gpu_prebacivanje_podataka = 0.0, gpu_zbrajanje = 0.0, gpu_redukcija_jedan_blok = 0.0, gpu_redukcija_8_blokova = 0.0, gpu_redukcija_jedan_stupac = 0.0, cpu = 0.0;


	if(argc != 3)
	{
		fprintf(stderr, "Za pokretanje %s program ocekuje 2 inta za dimenzije matrica, te 3 stringa koji oznacavaju imena datoteka iz kojih citamo, pisemo\n", argv[0]);
		return 0;
	}


	N = atoi(argv[1]);
	size = N*N;

    if(N > 8192)
    {
        fprintf(stderr, "Imamo vise od 8192 redka u matrici, pa nije moguce sve te podatke spremiti u konstantnu memoriju (konst mem 64kB).\n");
        return 0;
    }

	///Cilj je sto bolje optimizirati broj dretvi u odnosu na red matrice, tj. iskoristiti odgovarajuc broj blokova za redak matrice tako da
	/// imamo sta manji visak dretvi koje "ne rade nista".

    broj_dretvi_po_bloku = nadi_odgovarajuci_broj_dretvi(N);

    broj_blokova_po_redku = 1;
    while(broj_blokova_po_redku * broj_dretvi_po_bloku < N)
        broj_blokova_po_redku++;

    prosireni_size = broj_dretvi_po_bloku*broj_blokova_po_redku*N;

    printf("Broj dretvi po bloku je %d\n", broj_dretvi_po_bloku);
    printf("Broj blokova po redku je %d\n", broj_blokova_po_redku);

    gridDim = dim3(broj_blokova_po_redku, N, 1);
    blockDim = dim3(broj_dretvi_po_bloku, 1, 1);

    open_file(fA, argv[2], "r");

    // alokacija memorije na hostu
    host_alloc(hst_A, double, size);
    host_alloc(hst_A_prosirena, double, prosireni_size);
    host_alloc(hst_dijagonala, double, N);

    // alokacija memorije na deviceu
    cuda_exec(cudaMalloc(&dev_A, prosireni_size * sizeof(double)));

    // citanje podataka iz binarne datoteke
    read_file(hst_A, sizeof(double), size, fA);

    cpu -= timer();
    izracunaj_normalno(hst_A, N);
    cpu += timer();

    for(int i=0; i<N; i++)
    {
    	
    	hst_dijagonala[i] = (hst_A[i*N+i]); 
        //printf("i=%d, a[i]=%lg dijag[i]=%lg\n", i, hst_A[i*N+i], hst_dijagonala[i]);
    }
	prosiri_matricu(hst_A, hst_A_prosirena, N, broj_dretvi_po_bloku*broj_blokova_po_redku);


    gpu_prebacivanje_podataka -= timer();
    // kopiranje podataka na device
    cuda_exec(cudaMemcpy(dev_A, hst_A_prosirena, prosireni_size * sizeof(double), cudaMemcpyHostToDevice));
    cuda_exec(cudaMemcpyToSymbol(dijagonala, hst_dijagonala, N*sizeof(double)));   
    gpu_prebacivanje_podataka += timer();

    gpu_zbrajanje -= timer();
    zbrajanje<<<gridDim, blockDim>>>(dev_A, N);
    cuda_exec(cudaDeviceSynchronize());
    gpu_zbrajanje -= timer();

    gpu_redukcija_jedan_blok -= timer();
    switch (broj_dretvi_po_bloku) 
    {
		case 1024:
			reduction_jedan_blok<1024><<<gridDim, blockDim>>>(dev_A, prosireni_size);
			break;
		case  512:
			reduction_jedan_blok< 512><<<gridDim, blockDim>>>(dev_A, prosireni_size);
			break;
		case  256: 
			reduction_jedan_blok< 256><<<gridDim, blockDim>>>(dev_A, prosireni_size);
			break;
		case  128: 
			reduction_jedan_blok< 128><<<gridDim, blockDim>>>(dev_A, prosireni_size);
			break;
		case   64: 
			reduction_jedan_blok<  64><<<gridDim, blockDim>>>(dev_A, prosireni_size);
			break;
		case   32: 
			reduction_jedan_blok<  32><<<gridDim, blockDim>>>(dev_A, prosireni_size);
			break;
	}

    gpu_redukcija_jedan_blok += timer();

    /*gpu_redukcija_8_blokova -= timer();
    switch (broj_dretvi_po_bloku) {
		case 1024:
			reduction_8_blokova<1024><<<, blockDim>>>(dev_A, prosireni_size);
			break;
		case  512:
			reduction_8_blokova< 512><<<, blockDim>>>(dev_A, prosireni_size);
			break;
		case  256: 
			reduction_8_blokova< 256><<<, blockDim>>>(dev_A, prosireni_size);
			break;
		case  128: 
			reduction_8_blokova< 128><<<, blockDim>>>(dev_A, prosireni_size);
			break;
		case   64: 
			reduction_8_blokova<  64><<<, blockDim>>>(dev_A, prosireni_size);
			break;
		case   32: 
			reduction_8_blokova<  32><<<, blockDim>>>(dev_A, prosireni_size);
			break;
	}

    gpu_redukcija_8_blokova += timer();*/

    /*gpu_redukcija_jedan_stupac -= timer();
    switch (broj_dretvi_po_bloku) {
		case 1024:
			redukcija_jedan_stupac<1024><<<gridDim.y, blockDim>>>(dev_A, prosireni_size);
			break;
		case  512:
			redukcija_jedan_stupac< 512><<<gridDim.y, blockDim>>>(dev_A, prosireni_size);
			break;
		case  256: 
			redukcija_jedan_stupac< 256><<<gridDim.y, blockDim>>>(dev_A, prosireni_size);
			break;
		case  128: 
			redukcija_jedan_stupac< 128><<<gridDim.y, blockDim>>>(dev_A, prosireni_size);
			break;
		case   64: 
			redukcija_jedan_stupac<  64><<<gridDim.y, blockDim>>>(dev_A, prosireni_size);
			break;
		case   32: 
			redukcija_jedan_stupac<  32><<<gridDim.y, blockDim>>>(dev_A, prosireni_size);
			break;
	}

    gpu_redukcija_jedan_stupac += timer();*/



    gpu_prebacivanje_podataka -= timer();
    //kopiranje podatak na host
    cuda_exec(cudaMemcpy(hst_A_prosirena, dev_A, prosireni_size * sizeof(double), cudaMemcpyDeviceToHost));
    cuda_exec(cudaMemcpyFromSymbol(&norma2, result2, sizeof(double)));
    gpu_prebacivanje_podataka += timer();


    //to treba izbrisati
    //ispis_matrice(hst_A_prosirena, broj_dretvi_po_bloku*broj_blokova_po_redku, N);
    //ispis_matrice(hst_A, N, N);
    
    printf("GPU zbrajanje time: %#.3lgs\n", gpu_zbrajanje);
    printf("GPU redukcija jedan blok: %#.3lgs\n", gpu_redukcija_jedan_blok);
    //printf("GPU redukcija 8 blokova: %#.3lgs\n", gpu_redukcija_8_blokova);
    //printf("GPU redukcija jedan stupac: %#.3lgs\n", gpu_redukcija_jedan_stupac);
    printf("CPU execution time: %#.3lgs\n", cpu);
    printf("Norma pomocu GPU je: %lg\n", norma2);


    close_file(fA);printf("Nesto\n");
    host_free(hst_A);printf("Nesto\n");
    host_free(hst_A_prosirena);
    cuda_exec(cudaFree(dev_A));printf("Nesto\n");
    

	return 0;
}

