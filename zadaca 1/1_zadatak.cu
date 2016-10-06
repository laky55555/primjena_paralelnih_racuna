#include <stdio.h>
#include "cuda_wrapper.h"
#include <float.h>

#define broj_dretvi 512


void izracunaj_normalno(double *a, double *b, double *c, int m, int n)
{
    for(int i=0; i<m*n; i++)
        c[i] = a[i] * b[i];
}

void jesu_li_jednake(double *a, double *b, int ukupna_velicina)
{
    for(int i=0; i<ukupna_velicina; i++)
        if(abs(a[i] - b[i]) >= DBL_EPSILON)
        {
            printf("Prvo mjesto razlike je %d, a brojevi su: %lg i %lg\n", i, a[i], b[i]);
            return;
	}
    printf("Mnozenje je ispravno.\n");
}

__global__ void hadamardov_produkt (double *a, double *b, double *c, int ukupna_velicina)
{
    unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x;                
    unsigned int threadId = blockId * blockDim.x + threadIdx.x; 

    if (threadId<ukupna_velicina)
        c[threadId] = a[threadId] * b[threadId];

}


int main(int argc, char **argv)
{

    dim3 grid_dim, block_dim;
	int M, N, ukupna_velicina;
	double *hst_A, *hst_B, *hst_C, *hst_C_normalno;
	double *dev_A, *dev_B, *dev_C;
	FILE *fA, *fB, *fC;

    double gpu_racun = 0.0, gpu_ukupno = 0.0, cpu = 0.0;


	if(argc != 6)
	{
		fprintf(stderr, "Za pokretanje %s program ocekuje 2 inta za dimenzije matrica, te 3 stringa koji oznacavaju imena datoteka iz kojih citamo, pisemo\n", argv[0]);
		return 0;
	}


	M = atoi(argv[1]);
	N = atoi(argv[2]);
	ukupna_velicina = M*N;

    block_dim.x = broj_dretvi;
    
    grid_dim.x = min((ukupna_velicina + block_dim.x - 1) / block_dim.x, 65535);
    
    grid_dim.y = 1;
    while(ukupna_velicina > grid_dim.x*grid_dim.y*broj_dretvi)
        grid_dim.y += 1;
    
    printf("Dimenzije su: grid(%d, %d), blok(%d, %d).\n", grid_dim.x, grid_dim.y, block_dim.x, block_dim.y);

    open_file(fA, argv[3], "r");
    open_file(fB, argv[4], "r");
    open_file(fC, argv[5], "w");

    // alokacija memorije na hostu
    host_alloc(hst_A, double, ukupna_velicina);
    host_alloc(hst_B, double, ukupna_velicina);
    host_alloc(hst_C, double, ukupna_velicina);
    host_alloc(hst_C_normalno, double, ukupna_velicina);

    // alokacija memorije na deviceu
    cuda_exec(cudaMalloc(&dev_A, ukupna_velicina * sizeof(double)));
    cuda_exec(cudaMalloc(&dev_B, ukupna_velicina * sizeof(double)));
    cuda_exec(cudaMalloc(&dev_C, ukupna_velicina * sizeof(double)));

    // citanje podataka iz binarne datoteke
    read_file(hst_A, sizeof(double), ukupna_velicina, fA);
    read_file(hst_B, sizeof(double), ukupna_velicina, fB);

    gpu_ukupno -= timer();
    // kopiranje podataka na device
    cuda_exec(cudaMemcpy(dev_A, hst_A, ukupna_velicina * sizeof(double), cudaMemcpyHostToDevice));
    cuda_exec(cudaMemcpy(dev_B, hst_B, ukupna_velicina * sizeof(double), cudaMemcpyHostToDevice));
    gpu_ukupno += timer();

    gpu_racun -= timer();
    // pozivanje kernela
    hadamardov_produkt<<<grid_dim, block_dim>>>(dev_A, dev_B, dev_C, ukupna_velicina);
    cuda_exec(cudaDeviceSynchronize());

    gpu_racun += timer();

    gpu_ukupno -= timer();
    //kopiranje podatak na host
    cuda_exec(cudaMemcpy(hst_C, dev_C, ukupna_velicina * sizeof(double), cudaMemcpyDeviceToHost));
    gpu_ukupno += timer();
    gpu_ukupno += gpu_racun;

    cpu -= timer();
    izracunaj_normalno(hst_A, hst_B, hst_C_normalno, M, N);
    cpu += timer();

    printf("GPU racun time: %#.3lgs\n", gpu_racun);
    printf("GPU racun + prebacivanje: %#.3lgs\n", gpu_ukupno);
    printf("CPU execution time: %#.3lgs\n", cpu);

    jesu_li_jednake(hst_C, hst_C_normalno, ukupna_velicina);

    write_file(hst_C, sizeof(double), ukupna_velicina, fC);


    close_file(fA); close_file(fB); close_file(fC);
    host_free(hst_A); host_free(hst_B); host_free(hst_C); host_free(hst_C_normalno);
    cuda_exec(cudaFree(dev_A)); cuda_exec(cudaFree(dev_B)); cuda_exec(cudaFree(dev_C));


	return 0;
}
