#include  "cuda_wrapper.h"
#include <stdio.h>


//on 48kB shared memory max BLOCK_SIZE for double values = 512
//512 (threads) * 2 (2 matrices per thread) *
//5 (each thread deals with matrix padded for less bank conflicts) *
//8 (size of double) = 1024 * 5 * 8B = 40kB < 48kB
#define BLOCK_SIZE 512
//value of BLOCK_SIZE * 2, for reduction, so every thread has some work to do.
#define BLOCK_SIZE_X2 1024
//if two numbers differ in more than DIFFERENCE we call it different.
#define DIFFERENCE 0.001


/**
 * Auxiliary function for printing 8x8 3-diagonal matrix from 3 vectors.
 * First vector is diagonal, and other 2 sub-diagonal vectors.
 * Function accepts
 * @param a         Diagonal vector, dimension n.
 * @param b         Upper diagonal, dimension n-1.
 * @param c         Lower diagonal, dimension n-1.
 * @param dimension Dimension of diagonal vector/matrix.
 */
void print_matrix(double *a, double *b, double *c, int dimension)
{
	int min	= dimension < 8 ? dimension : 8;
	for(int i = 0; i < min; i++)
	{
		for(int j = 0; j < min; j++)
		{
			if (i == j)
				printf("%.2lg ", a[i]);
			else if(i == j+1)
				printf("%.2lg ", c[i-1]);
			else if(j == i+1)
				printf("%.2lg ", b[i]);
			else
				printf("0.00 ");
		}
		printf("\n");
	}
}

/**
 * Function for multiplying 2 matrix from same array. After calculating
 * product, product is saved in place of matrix with index left. Matrix with index
 * left is multiplied with matrix with index right L*R.
 * @param matrix       2D array in which both matrix are saved, and in which we will save product.
 * @param left_matrix  Index of left matrix.
 * @param right_matrix Index of right matrix.
 */
__device__ void matrix_multiplication(double matrix[][5], int left_matrix, int right_matrix)
{
	double temp[2];

	temp[0] = matrix[left_matrix][0] * matrix[right_matrix][0] + matrix[left_matrix][1] * matrix[right_matrix][2];
	temp[1] = matrix[left_matrix][0] * matrix[right_matrix][1] + matrix[left_matrix][1] * matrix[right_matrix][3];

	matrix[left_matrix][0] = temp[0];
	matrix[left_matrix][1] = temp[1];

	temp[0] = matrix[left_matrix][2] * matrix[right_matrix][0] + matrix[left_matrix][3] * matrix[right_matrix][2];
	temp[1] = matrix[left_matrix][2] * matrix[right_matrix][1] + matrix[left_matrix][3] * matrix[right_matrix][3];

	matrix[left_matrix][2] = temp[0];
	matrix[left_matrix][3] = temp[1];
}


/**
 * Function for multiplying 2 matrix from 2 different array. After calculating
 * product, product is saved in place of matrix with index left, in array from first
 * parameter. Matrix with index left is multiplied with matrix with index right L*R.
 * @param matrix_l     Array in which left matrix is saved, and in which we will save product.
 * @param matrix_r     2D array in which right matrix is saved.
 * @param left_matrix  Index of left matrix.
 * @param right_matrix Index of right matrix.
 */
__device__ void matrix_multiplication2(double *matrix_l, double matrix_r[][5], int left_matrix, int right_matrix)
{
	double temp[2];

	temp[0] = matrix_l[left_matrix] * matrix_r[right_matrix][0] + matrix_l[left_matrix + 1] * matrix_r[right_matrix][2];
	temp[1] = matrix_l[left_matrix] * matrix_r[right_matrix][1] + matrix_l[left_matrix + 1] * matrix_r[right_matrix][3];
	matrix_l[left_matrix] = temp[0];
	matrix_l[left_matrix + 1] = temp[1];

	temp[0] = matrix_l[left_matrix + 2] * matrix_r[right_matrix][0] + matrix_l[left_matrix + 3] * matrix_r[right_matrix][2];
	temp[1] = matrix_l[left_matrix + 2] * matrix_r[right_matrix][1] + matrix_l[left_matrix + 3] * matrix_r[right_matrix][3];
	matrix_l[left_matrix + 2] = temp[0];
	matrix_l[left_matrix + 3] = temp[1];
}



/**
 * Auxiliary function for calculating scan.
 * Function accepts matrices in 2D array and does reduction.
 * For true reduction (last element has all predecessors), second
 * argument needs to be power of 2.
 * At the end of execution, last element has all predecessors.
 * @param  matrix          2D array on which are we doing reduction.
 * @param  num_of_elements Number of elements in an array.
 * @return                 Returns log2(num_of_elements) for down_tree phase.
 */
__device__ int up_tree(double matrix[][5], int num_of_elements)
{
	int offset = 1;
	for (int d = num_of_elements >> 1; d > 0; d >>= 1)
	{
		__syncthreads();

		if (threadIdx.x < d)
		{
			int index_a = offset * (2 * threadIdx.x + 1) - 1;
			int index_b = offset * (2 * threadIdx.x + 2) - 1;

			matrix_multiplication(matrix, index_b, index_a);
		}

		offset <<= 1;
	}

	return offset;
}


/**
 * Auxiliary function for calculating scan.
 * Function accepts matrices in 2D array and does down-sweep,
 * function update all values from array so they are scan from all
 * their predecessors.
 * @param matrix 2D array on which are we doing reduction.
 * @param offset log2(number of elements in array), needed for knowing how
 *               many iteration we need until all elements are updated.
 */
__device__ void down_tree(double matrix[][5], int offset)
{
	int stride = 1;
	for (int working = 1; offset > 2; working += stride)
	{

		offset >>= 1;
		__syncthreads();
		//if(threadIdx.x == 0)
			//printf("Working = %d, offset = %d, stride = %d\n", working, offset, stride);

		if (threadIdx.x < working)
		{
			int index_a = offset * (threadIdx.x + 1) - 1;
			int index_b = index_a + offset / 2;
			matrix_multiplication(matrix, index_b, index_a);

		}

		stride <<= 1;
	}

}


/**
 * Function for calculating in place scan on given 2D array with length given from
 * second parameter. Length must be power of 2 for correct calculation.
 * At the end of execution each element in array should be
 * prefix operation from all it's predecessors.
 * @param matrix          2D array on which we will calculate scan.
 * @param num_of_elements Number of elements on which we want to do scan. For correct
 *                        calculation must be power of 2.
 */
__device__ void scan(double matrix[][5], int num_of_elements)
{
	int offset = up_tree(matrix, num_of_elements);

	down_tree(matrix, offset);

	__syncthreads();

}

/**
 * Function for first phase of LU decomposition on 3-diagonal matrix without pivoting.
 * Function does prefix product on each block of matrices, and save those values in global
 * memory. Function first load each vector (diagonal, upper diagonal and lower diagonal) into
 * 2D array of shared memory for calculating parallel prefix.
 * After each matrix is loaded, function calculate prefix product on each block.
 * After calculating elements are copied into global memory for true/global scan (in all
 * except first block product of all previous matrices is missing).
 * @param a           Array which represent diagonal in given 3-diagonal matrix.
 * @param b           Array which represent upper diagonal in given 3-diagonal matrix.
 * @param c           Array which represent lower diagonal in given 3-diagonal matrix.
 * @param temp_matrix Array in which we will save temporal values for each matrix.
 * @param dim         Positive integer who represent dimension of matrix. For correct
 *                    calculation his residue from 2*BLOCK_SIZE needs to be power of 2.
 */
__global__ void block_scan(double *a, double *b, double *c, double *temp_matrix, const int dim)
{
	__shared__ double matrix[BLOCK_SIZE*2][5];

	//Each thread reed 2 elements so for each block we use block*2 elements.
	int global_index = threadIdx.x * 2 + blockIdx.x * BLOCK_SIZE_X2;

	double *shifted_temp_matrix  = blockIdx.x * BLOCK_SIZE_X2 * 4 + temp_matrix;

	//if(threadIdx.x == 0)
		//printf("block = %d, global_index = %d, dim = %d\n", blockIdx.x, global_index, dim);


	if(global_index < dim)
	{

		matrix[threadIdx.x*2][0] = a[global_index];
		if(global_index == 0)
			matrix[threadIdx.x*2][1] = 0;
		else
			matrix[threadIdx.x*2][1] = -b[global_index - 1] * c[global_index - 1];
		matrix[threadIdx.x*2][2] = 1;
		matrix[threadIdx.x*2][3] = 0;

		matrix[threadIdx.x*2 + 1][0] = a[global_index + 1];
		matrix[threadIdx.x*2 + 1][1] = -b[global_index] * c[global_index];
		matrix[threadIdx.x*2 + 1][2] = 1;
		matrix[threadIdx.x*2 + 1][3] = 0;


		int num_of_elements = BLOCK_SIZE_X2 * (blockIdx.x + 1) <= dim ? BLOCK_SIZE_X2 : dim - BLOCK_SIZE_X2 * blockIdx.x;


		//if(threadIdx.x == 0)
			//printf("block = %d num_of_elements = %d\n",blockIdx.x, num_of_elements);


		scan(matrix, num_of_elements);

		//if(threadIdx.x == 0)
			//printf("Kopiranje novi matrica u globalnu memoriju\n");



		shifted_temp_matrix[threadIdx.x*8] = matrix[threadIdx.x*2][0];
		shifted_temp_matrix[threadIdx.x*8 + 1] = matrix[threadIdx.x*2][1];
		shifted_temp_matrix[threadIdx.x*8 + 2] = matrix[threadIdx.x*2][2];
		shifted_temp_matrix[threadIdx.x*8 + 3] = matrix[threadIdx.x*2][3];
		shifted_temp_matrix[threadIdx.x*8 + 4] = matrix[threadIdx.x*2 + 1][0];
		shifted_temp_matrix[threadIdx.x*8 + 5] = matrix[threadIdx.x*2 + 1][1];
		shifted_temp_matrix[threadIdx.x*8 + 6] = matrix[threadIdx.x*2 + 1][2];
		shifted_temp_matrix[threadIdx.x*8 + 7] = matrix[threadIdx.x*2 + 1][3];

		__syncthreads();

	}

	/*if(threadIdx.x == 0 && blockIdx.x) {
		printf("Zavrsio kernel block_scan, block number = %d\n", blockIdx.x);
		int broj = BLOCK_SIZE-1;
		while (broj < dim) {
			printf("matrix[%d][0] = %lg, matrix[%d][1] = %lg, matrix[%d][2] = %lg, matrix[%d][3] = %lg\n", broj, temp_matrix[4*broj], broj, temp_matrix[4*broj+1], broj, temp_matrix[4*broj+2], broj, temp_matrix[4*broj+3]);
			broj += BLOCK_SIZE;
		}
		broj = dim-1;
		printf("matrix[%d][0] = %lg, matrix[%d][1] = %lg, matrix[%d][2] = %lg, matrix[%d][3] = %lg\n", broj, temp_matrix[4*broj], broj, temp_matrix[4*broj+1], broj, temp_matrix[4*broj+2], broj, temp_matrix[4*broj+3]);
	}*/
}



/**
 * Function for final phase of LU decomposition on 3-diagonal matrix without pivoting.
 * Function does prefix product on matrices who are prefix products from corresponding blocks,
 * and after that update all matrices with global prefix product and calculate diagonal
 * and lower diagonal in LU decomposition.
 * @param matrix_old       Array of values symbolizing matrices in LU decomposition.
 *                         Each matrix is prefix product from all his predecessors from same block.
 * @param l                Array in which we will save lower diagonal of L matrix in LU decomposition.
 * @param d                Array in which we will save diagonal of L matrix in LU decomposition.
 * @param c                Array which contain upper diagonal from started matrix.
 * @param number_of_blocks Number of block in which we did first phase of prefix product. Needs to be
 *                         lesser than BLOCK_SIZE and power of 2.
 * @param dimension        Positive integer who represent dimension of matrix. For correct
 *                         calculation his residue from 2*BLOCK_SIZE needs to be power of 2.
 */
__global__ void final_scan (double *matrix_old, double *l, double *d, double *c, int number_of_blocks, int dimension)
{

	//if(threadIdx.x == 0)
		//printf("Kernel 2. Prije pocetka number_of_blocks = %d\n", number_of_blocks);


	__shared__ double matrix[BLOCK_SIZE][5];

	//TODO: razmisliti da li sve dretve do number_of_blocks treba ili samo do koliko zapravo blokova ima
	if(threadIdx.x < number_of_blocks)
	{
		double *shifted_d  = threadIdx.x * BLOCK_SIZE_X2 + d;
		double *shifted_l  = threadIdx.x * BLOCK_SIZE_X2 + l;
		double *shifted_c  = threadIdx.x * BLOCK_SIZE_X2 + c;
		double *shifted_matrix  = threadIdx.x * BLOCK_SIZE_X2 * 4 + matrix_old;

		int last_matrix = dimension < (threadIdx.x + 1) * BLOCK_SIZE_X2 ? dimension * 4 - 4 : (threadIdx.x + 1) * BLOCK_SIZE_X2 * 4 - 4;
		//printf("Zadnja matrica dretve %d je %d, pozicija prvog elementa je: %d.\n", threadIdx.x, last_matrix/4, last_matrix);

		matrix[threadIdx.x][0] = matrix_old[last_matrix];
		matrix[threadIdx.x][1] = matrix_old[last_matrix + 1];
		matrix[threadIdx.x][2] = matrix_old[last_matrix + 2];
		matrix[threadIdx.x][3] = matrix_old[last_matrix + 3];


		scan(matrix, number_of_blocks);

		/*for (size_t i = 0; i < blockDim.x; i++) {
			if (threadIdx.x == i)
				printf("Last matrix in block %d = %lg  %lg  %lg  %lg\n", threadIdx.x, matrix[threadIdx.x][0], matrix[threadIdx.x][1], matrix[threadIdx.x][2], matrix[threadIdx.x][3]);
			__syncthreads();
		}*/

		int number_of_elements = dimension < (threadIdx.x + 1) * BLOCK_SIZE_X2 ? dimension - threadIdx.x * BLOCK_SIZE_X2 : BLOCK_SIZE_X2;
		number_of_elements = number_of_elements < 0 ? 0 : number_of_elements;

		if(threadIdx.x != 0)
		{
			int lower_indeks = threadIdx.x - 1;

			#pragma unroll
			for (size_t i = 0; i < number_of_elements; i++)
				matrix_multiplication2(shifted_matrix, matrix, i*4, lower_indeks);

		}

		//TODO: to bi mozda trebalo odraditi s proslim zajedno.

		#pragma unroll
		for (size_t i = 0; i < number_of_elements; i++) {
			shifted_d[i] = shifted_matrix[i*4] / shifted_matrix[i*4 + 2];
			shifted_l[i] = shifted_c[i] / shifted_d[i];
		}



	}

	//__syncthreads();
	//if(threadIdx.x == 0)
		//printf("Zavrsio kernel final_scan\n");




}



/**
 * Function for first phase of solving Lz = y where L is lower triangular matrix
 * with 1 on diagonal, and 0 on non lower-diagonal. Function does prefix product on
 * each block of matrices, and save those values in global memory.
 * Function load lower-diagonal and result (y) vectors in
 * 2D array of shared memory for calculating parallel prefix product.
 * After each matrix is loaded, function calculate prefix product on each block.
 * After calculating elements are copied into global memory for true/global scan (in all
 * except first block product of all previous matrices is missing).
 * @param l           Array which represent lower-diagonal in L matrix from LU decomposition.
 * @param y           Array which represent result of 3-diagonal system.
 * @param temp_matrix Array in which we will save temporal values for each matrix.
 * @param dim         Positive integer who represent dimension of matrix. For correct
 *                    calculation his residue from 2*BLOCK_SIZE needs to be power of 2.
 */
__global__ void block_lz_y(double *l, double *y, double *temp_matrix, const int dim)
{
	__shared__ double matrix[BLOCK_SIZE*2][5];

	int global_index = threadIdx.x * 2 + blockIdx.x * BLOCK_SIZE_X2;

	double *shifted_temp_matrix  = blockIdx.x * BLOCK_SIZE_X2 * 4 + temp_matrix;


	//if(threadIdx.x == 0)
		//printf("block_lz_y kernel: block = %d, global_index = %d, dim = %d\n", blockIdx.x, global_index, dim);


	//TODO: kod ucitavanja 2 matirce treba vidjeti do kud ide.
	if(global_index < dim)
	{

		if(global_index == 0)
			matrix[threadIdx.x*2][0] = 0;
		else
			matrix[threadIdx.x*2][0] = -l[global_index - 1];
		matrix[threadIdx.x*2][1] = y[global_index];
		matrix[threadIdx.x*2][2] = 0;
		matrix[threadIdx.x*2][3] = 1;

		matrix[threadIdx.x*2 + 1][0] = -l[global_index];
		matrix[threadIdx.x*2 + 1][1] = y[global_index + 1];
		matrix[threadIdx.x*2 + 1][2] = 0;
		matrix[threadIdx.x*2 + 1][3] = 1;

		int num_of_elements = BLOCK_SIZE_X2 * (blockIdx.x + 1) <= dim ? BLOCK_SIZE_X2 : dim - BLOCK_SIZE_X2 * blockIdx.x;

		scan(matrix, num_of_elements);

		shifted_temp_matrix[threadIdx.x*8] = matrix[threadIdx.x*2][0];
		shifted_temp_matrix[threadIdx.x*8 + 1] = matrix[threadIdx.x*2][1];
		shifted_temp_matrix[threadIdx.x*8 + 2] = matrix[threadIdx.x*2][2];
		shifted_temp_matrix[threadIdx.x*8 + 3] = matrix[threadIdx.x*2][3];
		shifted_temp_matrix[threadIdx.x*8 + 4] = matrix[threadIdx.x*2 + 1][0];
		shifted_temp_matrix[threadIdx.x*8 + 5] = matrix[threadIdx.x*2 + 1][1];
		shifted_temp_matrix[threadIdx.x*8 + 6] = matrix[threadIdx.x*2 + 1][2];
		shifted_temp_matrix[threadIdx.x*8 + 7] = matrix[threadIdx.x*2 + 1][3];

		__syncthreads();
	}

	/*if(threadIdx.x == 0 && blockIdx.x) {
		printf("block_lz_y kernel: Zavrsio kernel block_scan, block number = %d\n", blockIdx.x);
		int broj = BLOCK_SIZE-1;
		while (broj < dim) {
			printf("block_lz_y kernel: matrix[%d][0] = %lg, matrix[%d][1] = %lg, matrix[%d][2] = %lg, matrix[%d][3] = %lg\n", broj, temp_matrix[4*broj], broj, temp_matrix[4*broj+1], broj, temp_matrix[4*broj+2], broj, temp_matrix[4*broj+3]);
			broj += BLOCK_SIZE;
		}
		broj = dim-1;
		printf("block_lz_y kernel: matrix[%d][0] = %lg, matrix[%d][1] = %lg, matrix[%d][2] = %lg, matrix[%d][3] = %lg\n", broj, temp_matrix[4*broj], broj, temp_matrix[4*broj+1], broj, temp_matrix[4*broj+2], broj, temp_matrix[4*broj+3]);
	}*/
}



/**
 * Function for final phase of Lz = y.
 * Function does prefix product on matrices who are prefix products from
 * corresponding blocks, and after that update all matrices with global
 * prefix product. After each matrix contain true prefix product calculate z value.
 * @param matrix_old       Array of values symbolizing matrices in LU decomposition.
 *                         Each matrix is prefix product from all his predecessors from same block.
 * @param z                Array in which we will save result from Lz = y system.
 * @param number_of_blocks Number of block in which we did first phase of prefix product. Needs to be
 *                         lesser than BLOCK_SIZE and power of 2.
 * @param dimension        Positive integer who represent dimension of matrix. For correct
 *                         calculation his residue from 2*BLOCK_SIZE needs to be power of 2.
 */
__global__ void final_lz_y (double *matrix_old, double *z, int number_of_blocks, int dimension)
{

	//if(threadIdx.x == 0)
		//printf("final_lz_y kernel:. Prije pocetka number_of_blocks = %d\n", number_of_blocks);


	__shared__ double matrix[BLOCK_SIZE][5];


	//TODO: razmisliti da li sve dretve do number_of_blocks treba ili samo do koliko zapravo blokova ima
	if(threadIdx.x < number_of_blocks)
	{
		double *shifted_z = threadIdx.x * BLOCK_SIZE_X2 + z;
		double *shifted_matrix  = threadIdx.x * BLOCK_SIZE_X2 * 4 + matrix_old;

		int last_matrix = dimension < (threadIdx.x + 1) * BLOCK_SIZE_X2 ? dimension * 4 - 4 : (threadIdx.x + 1) * BLOCK_SIZE_X2 * 4 - 4;
		//printf("final_lz_y kernel: Zadnja matrica dretve %d je %d, pozicija prvog elementa je: %d.\n", threadIdx.x, last_matrix/4, last_matrix);
		matrix[threadIdx.x][0] = matrix_old[last_matrix];
		matrix[threadIdx.x][1] = matrix_old[last_matrix + 1];
		matrix[threadIdx.x][2] = matrix_old[last_matrix + 2];
		matrix[threadIdx.x][3] = matrix_old[last_matrix + 3];


		scan(matrix, number_of_blocks);

		/*for (size_t i = 0; i < blockDim.x; i++) {
			if (threadIdx.x == i)
				printf("final_lz_y kernel: Last matrix in block %d = %lg  %lg  %lg  %lg\n", threadIdx.x, matrix[threadIdx.x][0], matrix[threadIdx.x][1], matrix[threadIdx.x][2], matrix[threadIdx.x][3]);
			__syncthreads();
		}*/

		int number_of_elements = dimension < (threadIdx.x + 1) * BLOCK_SIZE_X2 ? dimension - threadIdx.x * BLOCK_SIZE_X2 : BLOCK_SIZE_X2;
		number_of_elements = number_of_elements < 0 ? 0 : number_of_elements;

		if(threadIdx.x != 0)
		{
			int lower_indeks = threadIdx.x - 1;

			#pragma unroll
			for (size_t i = 0; i < number_of_elements; i++)
				matrix_multiplication2(shifted_matrix, matrix, i*4, lower_indeks);

		}

		//TODO: to bi mozda trebalo odraditi s proslim zajedno.

		#pragma unroll
		for (size_t i = 0; i < number_of_elements; i++) {
			shifted_z[i] = shifted_matrix[i*4 + 1];
		}



	}

	//__syncthreads();
	//if(threadIdx.x == 0)
		//printf("final_lz_y kernel: Zavrsio kernel final_scan\n");




}


/**
 * Function for first phase of solving Ux = z where U is upper triangular matrix
 * with 0 on every non diagonal/upper-diagonal element. Function does prefix product on
 * each block of matrices, and save those values in global memory.
 * Function load diagonal, upper-diagonal and result (y) vectors in
 * 2D array of shared memory for calculating parallel prefix product.
 * After each matrix is loaded, function calculate prefix product on each block.
 * After calculating elements are copied into global memory for true/global scan (in all
 * except first block product of all previous matrices is missing).
 * @param d           Array which represent diagonal in U matrix from LU decomposition.
 * @param l           Array which represent upper-diagonal in U matrix from LU decomposition.
 * @param y           Array which represent result of Lz = y.
 * @param temp_matrix Array in which we will save temporal values for each matrix.
 * @param dim         Positive integer who represent dimension of matrix. For correct
 *                    calculation his residue from 2*BLOCK_SIZE needs to be power of 2.
 * @param dim_old     True dimension (maybe non power of 2) of matrix.
 *                    We need it to know when prefix product needs to start because
 *                    of changing order in calculating, we start from back.
 */
__global__ void block_Ux_z(double *d, double *b, double *z, double *temp_matrix, const int dim, const int dim_old)
{
	__shared__ double matrix[BLOCK_SIZE*2][5];

	//okrecemo indekse i krecemo od straga
	//int global_index = BLOCK_SIZE_X2 * (gridDim.x - blockIdx.x) - threadIdx.x*2 - 1;
	//int global_index_normal = threadIdx.x * 2 + blockIdx.x * BLOCK_SIZE_X2;
	int global_index = dim - 1 - threadIdx.x * 2 - blockIdx.x * BLOCK_SIZE_X2;


	double *shifted_temp_matrix  = blockIdx.x * BLOCK_SIZE_X2 * 4 + temp_matrix;

	//if(threadIdx.x == 0)
		//printf("block_Ux_z kernel: block = %d, global_index = %d, dim = %d, gridDim.x = %d\n", blockIdx.x, global_index, dim, gridDim.x);


	//TODO: kod ucitavanja 2 matirce treba vidjeti do kud ide.
	if(global_index >= 0)
	{

		if(global_index >= dim_old - 1) {
			matrix[threadIdx.x*2][0] = 1;
			matrix[threadIdx.x*2][1] = 0;
		} else {
			matrix[threadIdx.x*2][0] = -b[global_index]/d[global_index];
			matrix[threadIdx.x*2][1] = z[global_index]/d[global_index];
		}
		matrix[threadIdx.x*2][2] = 0;
		matrix[threadIdx.x*2][3] = 1;

		if(global_index >= dim_old) {
			matrix[threadIdx.x*2 + 1][0] = 1;
			matrix[threadIdx.x*2 + 1][1] = 0;
		} else {
			matrix[threadIdx.x*2 + 1][0] = -b[global_index - 1]/d[global_index - 1];
			matrix[threadIdx.x*2 + 1][1] = z[global_index - 1]/d[global_index - 1];
		}
		matrix[threadIdx.x*2 + 1][2] = 0;
		matrix[threadIdx.x*2 + 1][3] = 1;

		/*for (int i = 0; i < blockDim.x; i++) {
			if(threadIdx.x == i) {
				printf("blockIdx.x = %d, Gl indx = %d, Matrica %d => %lg %lg %lg %lg\n",blockIdx.x, global_index, i*2, matrix[i*2][0], matrix[i*2][1], matrix[i*2][2], matrix[i*2][3]);
				printf("blockIdx.x = %d, Gl indx = %d, Matrica %d => %lg %lg %lg %lg\n",blockIdx.x, global_index, i*2 + 1, matrix[i*2 + 1][0], matrix[i*2 + 1][1], matrix[i*2 + 1][2], matrix[i*2 + 1][3]);
			}
			__syncthreads();
		}*/

		int num_of_elements = BLOCK_SIZE_X2 * (blockIdx.x + 1) <= dim ? BLOCK_SIZE_X2 : dim - BLOCK_SIZE_X2 * blockIdx.x;

		//if(threadIdx.x == 0)
			//printf("block_Ux_z kernel: block = %d num_of_elements = %d\n",blockIdx.x, num_of_elements);

		scan(matrix, num_of_elements);


		shifted_temp_matrix[threadIdx.x*8] = matrix[threadIdx.x*2][0];
		shifted_temp_matrix[threadIdx.x*8 + 1] = matrix[threadIdx.x*2][1];
		shifted_temp_matrix[threadIdx.x*8 + 2] = matrix[threadIdx.x*2][2];
		shifted_temp_matrix[threadIdx.x*8 + 3] = matrix[threadIdx.x*2][3];
		shifted_temp_matrix[threadIdx.x*8 + 4] = matrix[threadIdx.x*2 + 1][0];
		shifted_temp_matrix[threadIdx.x*8 + 5] = matrix[threadIdx.x*2 + 1][1];
		shifted_temp_matrix[threadIdx.x*8 + 6] = matrix[threadIdx.x*2 + 1][2];
		shifted_temp_matrix[threadIdx.x*8 + 7] = matrix[threadIdx.x*2 + 1][3];

		/*for (int i = 0; i < blockDim.x; i++) {
			if(threadIdx.x == i) {
				printf("Gl indx = %d, Matrica %d => %lg %lg %lg %lg\n",global_index, i*2, matrix[i*2][0], matrix[i*2][1], matrix[i*2][2], matrix[i*2][3]);
				printf("Gl indx = %d, Matrica %d => %lg %lg %lg %lg\n",global_index, i*2 + 1, matrix[i*2 + 1][0], matrix[i*2 + 1][1], matrix[i*2 + 1][2], matrix[i*2 + 1][3]);
			}
			__syncthreads();
		}*/

		//__syncthreads();

	}


	/*f(threadIdx.x == 0 && blockIdx.x) {
		printf("block_Ux_z kernel: Zavrsio kernel block_scan, block number = %d\n", blockIdx.x);
		int broj = BLOCK_SIZE-1;
		while (broj < dim) {
			printf("block_Ux_z kernel: matrix[%d][0] = %lg, matrix[%d][1] = %lg, matrix[%d][2] = %lg, matrix[%d][3] = %lg\n", broj, temp_matrix[4*broj], broj, temp_matrix[4*broj+1], broj, temp_matrix[4*broj+2], broj, temp_matrix[4*broj+3]);
			broj += BLOCK_SIZE;
		}
		broj = dim-1;
		printf("block_Ux_z kernel: matrix[%d][0] = %lg, matrix[%d][1] = %lg, matrix[%d][2] = %lg, matrix[%d][3] = %lg\n", broj, temp_matrix[4*broj], broj, temp_matrix[4*broj+1], broj, temp_matrix[4*broj+2], broj, temp_matrix[4*broj+3]);
	}*/
}


/**
 * Function for final phase of Ux = z.
 * Function does prefix product on matrices who are prefix products from
 * corresponding blocks, and after that update all matrices with global
 * prefix product. After each matrix contain true prefix product calculate x value.
 * @param matrix_old       Array of values symbolizing matrices in LU decomposition.
 *                         Each matrix is prefix product from all his predecessors from same block.
 * @param z                Array in which we will save result from Ux = z system.
 *                         We also need zn, true (maybe non power of 2) last element in vector z.
 * @param d                We need dn, true (maybe non power of 2) last element in vector d.
 * @param number_of_blocks Number of block in which we did first phase of prefix product. Needs to be
 *                         lesser than BLOCK_SIZE and power of 2.
 * @param dimension        Positive integer who represent dimension of matrix. For correct
 *                         calculation his residue from 2*BLOCK_SIZE needs to be power of 2.
 * @param dim_old          True dimension (maybe non power of 2) of matrix.
 *                         We need it to know who is true last element in vectors.
 */
__global__ void final_Ux_z (double *matrix_old, double *z, double *d, int number_of_blocks, int dimension, int dim_old)
{

	__shared__ double matrix[BLOCK_SIZE][5];

	double zn = z[dim_old - 1];
	double dn = d[dim_old - 1];
	double xn = zn / dn;


	//TODO: razmisliti da li sve dretve do number_of_blocks treba ili samo do koliko zapravo blokova ima
	if(threadIdx.x < number_of_blocks)
	{
		double *shifted_z = threadIdx.x * BLOCK_SIZE_X2 + z;
		double *shifted_matrix  = threadIdx.x * BLOCK_SIZE_X2 * 4 + matrix_old;
		double *backward_matrix  = (number_of_blocks - 1 - threadIdx.x) * BLOCK_SIZE_X2 * 4 + matrix_old;

		int last_matrix = dimension < (threadIdx.x + 1) * BLOCK_SIZE_X2 ? dimension * 4 - 4 : (threadIdx.x + 1) * BLOCK_SIZE_X2 * 4 - 4;
		//printf("final_Ux_z kernel: Zadnja matrica dretve %d je %d, pozicija prvog elementa je: %d.\n", threadIdx.x, last_matrix/4, last_matrix);
		matrix[threadIdx.x][0] = matrix_old[last_matrix];
		matrix[threadIdx.x][1] = matrix_old[last_matrix + 1];
		matrix[threadIdx.x][2] = matrix_old[last_matrix + 2];
		matrix[threadIdx.x][3] = matrix_old[last_matrix + 3];

		scan(matrix, number_of_blocks);


		/*for (size_t i = 0; i < blockDim.x; i++) {
			if (threadIdx.x == i)
				printf("final_Ux_z kernel: Last matrix in block %d = %lg  %lg  %lg  %lg\n", threadIdx.x, matrix[threadIdx.x][0], matrix[threadIdx.x][1], matrix[threadIdx.x][2], matrix[threadIdx.x][3]);
			__syncthreads();
		}*/


		int number_of_elements = dimension < (threadIdx.x + 1) * BLOCK_SIZE_X2 ? dimension - threadIdx.x * BLOCK_SIZE_X2 : BLOCK_SIZE_X2;
		number_of_elements = number_of_elements < 0 ? 0 : number_of_elements;
		//number_of_elements = number_of_elements < dim_old - BLOCK_SIZE_X2 * threadIdx.x ? number_of_elements : dim_old;
		number_of_elements = number_of_elements < dim_old - BLOCK_SIZE_X2 * threadIdx.x ? number_of_elements : dim_old - BLOCK_SIZE_X2 * threadIdx.x;
		//printf("threadIdx.x = %d, number_of_elements = %d\n", threadIdx.x, number_of_elements);

		if(threadIdx.x != 0)
		{
			int lower_indeks = threadIdx.x - 1;

			#pragma unroll
			for (size_t i = 0; i < number_of_elements; i++) {
				matrix_multiplication2(shifted_matrix, matrix, i*4, lower_indeks);
			}

		}


		/*for (int i = 0; i < blockDim.x; i++) {
			if(threadIdx.x == i) {
				for(int j = 0; j < number_of_elements; j++) {
					int k = i*4*BLOCK_SIZE_X2 + j*4;
					printf("threadIdx.x = %d, Matrica %d, matrica_pozicija = %d => %lg %lg %lg %lg\n",threadIdx.x, j + BLOCK_SIZE_X2*threadIdx.x,
							k, matrix_old[k], matrix_old[k+1], matrix_old[k+2], matrix_old[k+3]);

				}
			}
			__syncthreads();
		}*/

		//TODO: to bi mozda trebalo odraditi s proslim zajedno.

		//number_of_elements = number_of_elements < dim_old - BLOCK_SIZE_X2 * threadIdx.x ? number_of_elements : dim_old - BLOCK_SIZE_X2 * threadIdx.x;

		#pragma unroll
		for (int i = 0, j = number_of_elements - 1; i < number_of_elements; i++, j--) {
			//if(threadIdx.x == 0 && i == 0)
				//printf("final_Ux_z kernel: Usao u pragma unrool 2, z[0] = %lg \n", z[0]);
			//shifted_z[i] = shifted_matrix[i*4 + 1];
			//shifted_z[i] = backward_matrix[j*4] * xn + backward_matrix[j*4 + 1];
			//zadnja pa za nazad
			z[i+ threadIdx.x * BLOCK_SIZE_X2] =
				matrix_old[(dimension - 1)*4 - i*4 - (threadIdx.x * BLOCK_SIZE_X2 * 4)] * xn +
				matrix_old[(dimension - 1)*4 - i*4 - (threadIdx.x * BLOCK_SIZE_X2 * 4) + 1];
			/*printf("z[%d] = matrix_old[%d] * xn + matrix_old[%d] = %lg * %lg + %lg = %lg;\n", i + threadIdx.x * BLOCK_SIZE_X2,
					(dimension - 1)*4 - i*4 - (threadIdx.x * BLOCK_SIZE_X2 * 4), (dimension - 1)*4 - i*4 - (threadIdx.x * BLOCK_SIZE_X2 * 4) + 1,
					matrix_old[(dim_old - 1)*4 - i*4 - (threadIdx.x * BLOCK_SIZE_X2 * 4)], xn,
					matrix_old[(dim_old - 1)*4 - i*4 - (threadIdx.x * BLOCK_SIZE_X2 * 4) + 1], z[i]);*/
		}



	}

	//if(threadIdx.x == 0)
		//printf("final_Ux_z kernel: Zavrsio kernel final_scan\n");


	__syncthreads();


}





/**
 * Auxiliary function for calculate LU decomposition without pivoting on host.
 * We are using it to check calculation on device.
 * @param N Dimensions of vectors/matrix.
 * @param a Diagonal in a matrix.
 * @param b Upper-diagonal in a matrix.
 * @param c Lower-diagonal in a matrix.
 * @param l Vector in which we will save l values (lower-diagonal) after LU decomposition
 * @param d Vector in which we will save d values (diagonal) after LU decomposition
 */
void lu_factorization_host(const int N, double *a, double *b, double *c, double *l, double *d)
{
	d[0] = a[0];
	for(int i = 1; i < N; i++)
	{
		d[i] = a[i] - ( (b[i-1] * c[i-1]) / d[i-1] );
		l[i-1] = c[i-1]/d[i-1];
	}
}

/**
 * Auxiliary function for calculate Lz = y system on host.
 * Matrix L is lower-triangular matrix with 1 on diagonal and 0 on
 * non sub-diagonal elements.
 * After execution in array z we will have result.
 * @param N dimension of matrix L.
 * @param l Array which contain lower-diagonal vector.
 * @param y Array with result of Lz => y.
 * @param z Array in which we will save solution.
 */
void lz_y_host(const int N, double *l, double *y, double *z)
{
	z[0] = y[0];
	for (size_t i = 1; i < N; i++)
		z[i] = y[i] - l[i-1] * z[i-1];

}

/**
 * Auxiliary function for calculate Ux = z system on host.
 * Matrix U is upper-triangular matrix with vector d on diagonal,
 * vector b on upper-diagonal and 0 rest.
 * After execution in array x we will have result.
 * @param x       [description]
 * @param d       [description]
 * @param b       [description]
 * @param z       [description]
 * @param N dimension of matrix U.
 * @param x Array in which we will save solution.
 * @param d Array which contain diagonal vector.
 * @param b Array which contain upper-diagonal vector.
 * @param z Array which contain vector z from system Ux = z.
 */
void ux_z_host(const int N, double *x, double *d, double *b, double *z)
{
	x[N - 1] = z[N - 1] / d[N - 1];
	//printf("\nxn = %lg\n\n", x[N-1]);
	for (int i = N - 2; i >= 0; i--)
		x[i] = -b[i]/d[i]*x[i+1] + z[i]/d[i];

}

/**
 * Auxiliary function for checking if given vectors differ in more than DIFFERENCE.
 * Function prints all positions on which vectors differ and values that they have.
 * @param dim Dimension of arrays we want to check.
 * @param a   First vector.
 * @param b   Second vector.
 */
void check_vectors(int dim, double *a, double *b)
{
	for (size_t i = 0; i < dim; i++) {
		if (abs(a[i] - b[i]) >= DIFFERENCE)
			printf("On place %d, a[%d] = %lg, b[%d] = %lg\n", i, i, a[i], i, b[i]);

	}
}


/**
 * Function for calculating first number greater or equal to number in first argument
 * that when we calculate residue from that number and number in second argument
 * we get power of 2.
 * @param  number             Number which we will increase.
 * @param  number_of_elements Test number for calculating reminder.
 * @return                    Integer whose residue with number_of_elements is 0.
 */
int first_power_2(int number, int number_of_elements)
{
	int residue = number % number_of_elements;
	if(residue == 0)
		return number;

	int pow2_n = 1;
	while (residue > pow2_n)
		pow2_n <<= 1;

	return (number / number_of_elements) * number_of_elements + pow2_n;
}

void print_vector(const int N, double *a)
{
	for (size_t i = 0; i < N; i++)
		printf("%lg ", a[i]);

	printf("\n");
}


/**
 * Program accepts dimension value for vector dimension n (n>0 natural number), and
 * paths for 4 vectors (two n dimensional, and two n-1 dimensional),
 * and solves using GPU given 3-diagonal system of equations using parallel prefix.
 * Program also solves 3-diagonal system using CPU for testing purpose.
 * First parameter is lower diagonal, second diagonal, third upper diagonal, and
 * fourth value y of equation Ax = y.
 * @param  argc Numbers of strings given over command line, expecting 5 strings.
 * @param  argv Array of strings.
 * @return      Returns 0 if everything is OK.
 */
int main(int argc, char const *argv[]) {

	//Program needs 6 parameters for functioning correctly.
	if(argc != 6)
	{
		fprintf(stderr, "Program %s expects positive integer for dimension, and vectors of real numbers with following dimensions: n-1, n, n-1, n.\n", argv[0]);
		return -1;
	}

	//We use BLOCK_SIZE * 2 * 4 double values of share memory so we need it bigger than L1 cache.
	cuda_exec(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

	double	cpu_time = 0.0;
	double	gpu_time = 0.0;
	int dimension, dimension_old;
	dim3 grid_size, block_size;

	FILE *fA, *fB, *fC, *fY;

	//Arrays in host memory for reading data and checking calculations.
	double *hst_a;
	double *hst_b;
	double *hst_c;
	double *hst_y;
	double *hst_l;
	double *hst_d;
	double *hst_z;
	double *hst_x;

	//Copies of device vectors in host memory.
	double *hst_d_dev_copy;
	double *hst_l_dev_copy;
	double *hst_temp_matrix_dev_copy;
	double *hst_z_dev_copy;
	double *hst_x_dev_copy;

	//Arrays in device memory used for calculations.
	double *dev_a;
	double *dev_b;
	double *dev_c;
	double *dev_y;
	double *dev_d;
	double *dev_l;
	double *dev_temp_matrix;


	//True dimension of matrix.
	dimension_old = atoi(argv[1]);
	//For correct calculating scan we need each block to work with power of 2 elements.
	dimension = first_power_2(dimension_old, BLOCK_SIZE_X2);

	//Allocating memory on host.
	host_alloc(hst_a, double, dimension * sizeof(double));
	host_alloc(hst_b, double, dimension * sizeof(double));
	host_alloc(hst_c, double, dimension * sizeof(double));
	host_alloc(hst_y, double, dimension * sizeof(double));
	host_alloc(hst_l, double, dimension * sizeof(double));
	host_alloc(hst_d, double, dimension * sizeof(double));
	host_alloc(hst_z, double, dimension * sizeof(double));
	host_alloc(hst_x, double, dimension * sizeof(double));

	//Allocating memory on host used for copying values from device.
	host_alloc(hst_d_dev_copy, double, dimension * sizeof(double));
	host_alloc(hst_l_dev_copy, double, dimension * sizeof(double));
	host_alloc(hst_temp_matrix_dev_copy, double, dimension * 4 * sizeof(double));
	host_alloc(hst_z_dev_copy, double, dimension * sizeof(double));
	host_alloc(hst_x_dev_copy, double, dimension * sizeof(double));

	//Allocating memory on device.
	cuda_exec(cudaMalloc(&dev_a, dimension * sizeof(double)));
	cuda_exec(cudaMalloc(&dev_b, dimension * sizeof(double)));
	cuda_exec(cudaMalloc(&dev_c, dimension * sizeof(double)));
	cuda_exec(cudaMalloc(&dev_y, dimension * sizeof(double)));
	cuda_exec(cudaMalloc(&dev_d, dimension * sizeof(double)));
	cuda_exec(cudaMalloc(&dev_l, dimension * sizeof(double)));
	cuda_exec(cudaMalloc(&dev_temp_matrix, 4 * dimension * sizeof(double)));

	//Calculating sizes for blocks and grid on device.
	//Each block will do scan on BLOCK_SIZE * 2 elements => affects grid_size.
	block_size = dim3(BLOCK_SIZE, 1, 1);
	//Grid size must has enough blocks so number_of_blocks * BLOCK_SIZE * 2 > dimension
	grid_size.x = ((atoi(argv[1]) - 1) / 2) / BLOCK_SIZE + 1;;

	printf("Dimensions are: grid (%d, 0, 0), blok (%d, 0, 0), matrix is %dx%d, dimension_old = %d\n", grid_size.x, block_size.x, dimension, dimension, dimension_old);

	//If grid size is greater than 1024 we can not do final prefix product i one block.
	if (grid_size.x > 1024) {
		fprintf(stderr, "grid_size.x = %d, but must be under 1024.\n", grid_size.x);
		goto end;
	}

	//Opening and reading data from files.
	open_file(fA, argv[3], "r");
    open_file(fB, argv[2], "r");
    open_file(fC, argv[4], "r");
    open_file(fY, argv[5], "r");

	//Difference between dimension and dimension_old is filled with zeros.
	read_file(hst_a, sizeof(double), dimension_old, fA);
    read_file(hst_b, sizeof(double), dimension_old - 1, fB);
    read_file(hst_c, sizeof(double), dimension_old - 1, fC);
	read_file(hst_y, sizeof(double), dimension_old, fY);

	//Copying vectors from host to device memory.
	cuda_exec(cudaMemcpy(dev_a, hst_a, dimension * sizeof(double), cudaMemcpyHostToDevice));
	cuda_exec(cudaMemcpy(dev_b, hst_b, dimension * sizeof(double), cudaMemcpyHostToDevice));
	cuda_exec(cudaMemcpy(dev_c, hst_c, dimension * sizeof(double), cudaMemcpyHostToDevice));
	cuda_exec(cudaMemcpy(dev_y, hst_y, dimension * sizeof(double), cudaMemcpyHostToDevice));

	printf("Starting matrix:\n");
	print_matrix(hst_a, hst_b, hst_c, dimension);
	printf("\n");


	printf("Starting calculation on host.\n");
	cpu_time -= timer();
	lu_factorization_host(dimension, hst_a, hst_b, hst_c, hst_l, hst_d);
	lz_y_host(dimension, hst_l, hst_y, hst_z);
	ux_z_host(dimension_old, hst_x, hst_d, hst_b, hst_z);
	cpu_time += timer();
	printf("Finished calculation on host.\n");



	printf("Starting calculation on device.\n");
	gpu_time -= timer();

	printf("Starting block_scan kernel.\n");
	block_scan<<<grid_size, block_size>>>(dev_a, dev_b, dev_c, dev_temp_matrix, dimension);
	cudaDeviceSynchronize();

	printf("Starting final_scan kernel.\n");
	final_scan<<<1, block_size>>>(dev_temp_matrix, dev_l, dev_d, dev_c, first_power_2(grid_size.x, BLOCK_SIZE), dimension);
	cudaDeviceSynchronize();

	printf("Copying results on host and checking values.\n");
	cuda_exec(cudaMemcpy(hst_d_dev_copy, dev_d, dimension * sizeof(double), cudaMemcpyDeviceToHost));
	cuda_exec(cudaMemcpy(hst_l_dev_copy, dev_l, dimension * sizeof(double), cudaMemcpyDeviceToHost));

	printf("Check for difference between d_dev and d_host\n");
	check_vectors(dimension, hst_d_dev_copy, hst_d);
	printf("Check for difference between l_dev and l_host\n");
	check_vectors(dimension - 1, hst_l_dev_copy, hst_l);

	printf("LU matrix on device:\n");
	print_matrix(hst_d_dev_copy, hst_b, hst_l_dev_copy, dimension);
	printf("\n");


	printf("Starting block_lz_y kernel.\n");
	block_lz_y<<<grid_size, block_size>>>(dev_l, dev_y, dev_temp_matrix, dimension);
	cudaDeviceSynchronize();
	printf("Starting final_lz_y kernel.\n");
	final_lz_y<<<1, block_size>>>(dev_temp_matrix, dev_y, first_power_2(grid_size.x, BLOCK_SIZE), dimension);
	cudaDeviceSynchronize();


	printf("Copy and check for difference between z_dev and z_host\n");
	cuda_exec(cudaMemcpy(hst_z_dev_copy, dev_y, dimension * sizeof(double), cudaMemcpyDeviceToHost));
	check_vectors(dimension, hst_z_dev_copy, hst_z);


	printf("Vector z\n");
	print_vector(dimension, hst_z_dev_copy);
	printf("Vector d\n");
	print_vector(dimension, hst_d_dev_copy);

	printf("Starting block_Ux_z kernel.\n");
	block_Ux_z<<<grid_size, block_size>>>(dev_d, dev_b, dev_y, dev_temp_matrix, dimension, dimension_old);
	cudaDeviceSynchronize();

	printf("Matirix backward_matrix\n");
	cuda_exec(cudaMemcpy(hst_temp_matrix_dev_copy, dev_temp_matrix, dimension * 4 * sizeof(double), cudaMemcpyDeviceToHost));

	printf("Vector b\n");
	print_vector(dimension, hst_b);
	printf("Vector d\n");
	print_vector(dimension, hst_d);
	printf("Vector z\n");
	print_vector(dimension, hst_z);
	printf("matirx temp\n");
	/*for (size_t i = 0; i < dimension; i++) {
		printf("%lg %lg %lg %lg\n", hst_temp_matrix_dev_copy[4*i], hst_temp_matrix_dev_copy[4*i+1], hst_temp_matrix_dev_copy[4*i+2], hst_temp_matrix_dev_copy[4*i+3]);
	}*/

	printf("Starting final_Ux_z kernel.\n");
	final_Ux_z<<<1, block_size>>>(dev_temp_matrix, dev_y, dev_d, first_power_2(grid_size.x, BLOCK_SIZE), dimension, dimension_old);
	cudaDeviceSynchronize();

	gpu_time += timer();

	printf("Copy and check for difference between x_dev and x_host\n");
	cuda_exec(cudaMemcpy(hst_x_dev_copy, dev_y, dimension * sizeof(double), cudaMemcpyDeviceToHost));
	check_vectors(dimension_old, hst_x_dev_copy, hst_x);

	printf("CPU execution time: %#.3lgs\n", cpu_time);
	printf("GPU execution time: %#.3lgs\n", gpu_time);

//Deallocating all dynamically allocated memory.
end:

	free(hst_a);
	free(hst_b);
	free(hst_c);
	free(hst_y);
	free(hst_d);
	free(hst_l);
	free(hst_d_dev_copy);
	free(hst_l_dev_copy);
	free(hst_temp_matrix_dev_copy);
	free(hst_z);
	free(hst_z_dev_copy);

	cuda_exec(cudaFree(dev_a));
	cuda_exec(cudaFree(dev_b));
	cuda_exec(cudaFree(dev_c));
	cuda_exec(cudaFree(dev_y));
	cuda_exec(cudaFree(dev_d));
	cuda_exec(cudaFree(dev_l));
	cuda_exec(cudaFree(dev_temp_matrix));


	return 0;
}
