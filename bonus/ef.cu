#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <sstream>
#include "CImg.h"
#include  "cuda_wrapper.h"
#include "cublas_v2.h"
#include "magma.h"
#include "magma_lapack.h"

using namespace cimg_library;


/**
 * Auxiliary function for printing first few rows of matrix.
 * @param matrix Matrix we want to print.
 * @param m      Number of rows in matrix.
 * @param n      Number of columns in matrix.
 */
void print_matrix(float *matrix, int m, int n)
{
	for(int i = 0; i < 5; i++)
	{
		for(int j = 0; j < 10; j++)
			printf("%f ", matrix[i*n + j]);
		printf("\n");
	}
}

/**
 * Function for converting number to string.
 * @param  number Number to convert to string.
 * @return String Input number in string form.
 */
std::string number_to_string(int number)
{
	std::ostringstream convert;
	convert << number;
	return convert.str();
}


/**
 * Auxiliary function for loading picture data from memory to GPU memory.
 * Function also calculate mean value for each pixel of pictures and save
 * it on device on dev_mean
 * @param  directory  Parent directory in which we can find all pictures.
 * @param  pic_number Number of pictures we will load into memory
 * @param  pic_dim    Dimension of each picture. All pictures have the same resolution.
 * @param  hst_matrix Data from all pictures saved in host memory.
 * @param  dev_matrix Data from all pictures saved in device memory.
 * @param  dev_mean      Mean value for each pixel saved in device vector.
 * @param  handle     initialized handle for cublas operations.
 * @param  hst_target Data from picture we are looking for, copying from host to device.
 * @param  dev_target Data from picture we are looking for.
 * @return int        Returns 0 if everything is OK.
 */
int load_pictures(const char *directory, int pic_number, int pic_dim, float *hst_matrix, float *dev_matrix, float *dev_mean, cublasHandle_t handle, float *hst_target, float *dev_target)
{

	float alfa = 1./pic_number;

	for(int i = 0; i < pic_number; i++)
	{
		std::string pic_name(std::string (directory) + "/s" + number_to_string(i/10+1) + "/" + number_to_string(i%10+1) + ".pgm");
		CImg<float> picture(pic_name.c_str());
		for(int j = 0; j < pic_dim; j++)
			hst_matrix[i*pic_dim + j] = picture.data()[j];

		//Copying data onto device and accumulating vector sum in one vector; maybe it can be done in parallel.
		cuda_exec(cudaMemcpy(i*pic_dim + dev_matrix, i*pic_dim + hst_matrix, pic_dim * sizeof(float), cudaMemcpyHostToDevice));

		//Accumulating mean over each pixel in vector dev_mean.
		cublas_exec(cublasSaxpy(handle, pic_dim, &alfa, i*pic_dim + dev_matrix, 1, dev_mean, 1 ));

	}

	cuda_exec(cudaMemcpy(dev_target, hst_target, pic_dim * sizeof(float), cudaMemcpyHostToDevice));

	return 0;
}

/**
 * Function for calculating singular value decomposition (SVD).
 * Function gets and return all values through arguments.
 * 					M = U * S * V^T
 * http://icl.cs.utk.edu/projectsfiles/magma/doxygen/group__magma__gesvd.html#ga96e26a734d9c48e5c994863c4e2d83f1
 * @param  pic_dim     Number of rows in matrix for SVD.
 * @param  pic_number  Number of cols in matrix for SVD.
 * @param  matrix      Matrix for SVD.
 * @param  sing_values Rectangular diagonal matrix with non-negative real numbers on the diagonal.
 *                     The singular values of matrix, sorted so that S(i) >= S(i+1)
 * @param  U		   Unitary matrix.
 * @param  V           Unitary matrix.
 * @return int         Returns 0 if everything is OK.
 */
int calculating_svd(int pic_dim, int pic_number, float *matrix, float *sing_values, float *U, float *V)
{
	magma_init();

	int info;
	int lwork = -1;
	float *work;
	host_alloc(work, float, sizeof(float));

	//
	magma_sgesvd(MagmaAllVec, MagmaAllVec, pic_dim, pic_number, matrix, pic_dim, sing_values, U, pic_dim, V, pic_number, work, lwork, &info);
	lwork = work[0];
	host_alloc(work, float, lwork * sizeof(float));
	magma_sgesvd(MagmaAllVec, MagmaAllVec, pic_dim, pic_number, matrix, pic_dim, sing_values, U, pic_dim, V, pic_number, work, lwork, &info);

	magma_finalize();

	return 0;
}


/**
 * Program need 2 parameters, first is location of picture you want to find, and
 * second is directory location for training set from which we will find the closest
 * looking picture for one given in first parameter.
 * All input pictures need to be the same size and resolution. All pictures need
 * to be black and white.
 * @param  argc Number of parameters.
 * @param  argv Array of strings which hold parameters.
 * @return int 	Returns 0 if everything is OK.
 */
int main(int argc, char const *argv[])
{

	//Program needs 3 parameters for functioning correctly.
	if(argc != 3)
	{
		fprintf(stderr, "Program %s expects: picture location, directory with test pictures.\n", argv[0]);
		return -1;
	}

	//Loading target picture for checking pictures dimensions.
	CImg<float> target(argv[1]);

	//Number of pictures for training set.
	unsigned int pic_number = 400;
	unsigned int pic_width = target.width();
	unsigned int pic_height = target.height();
	unsigned int pic_dim = pic_height * pic_width;


	//Defining all needed pointers.
	float *hst_matrix;
	float *hst_norms;

	float *hst_sing_values;
	float *hst_u;
	float *hst_v;

	float *dev_target;
	float *dev_mean;
	float *dev_matrix;
	float *dev_u;
	float *dev_result_matrix;
	float *dev_result_target;
	float *dev_norms;

	cublasHandle_t handle;

	//Allocating host memory.
	host_alloc(hst_matrix, float, pic_dim * pic_number * sizeof(float));
	host_alloc(hst_norms, float, pic_number * sizeof(float));

	host_alloc(hst_sing_values, float, pic_number * sizeof(float));
	host_alloc(hst_u, float, pic_dim * pic_dim * sizeof(float));
	host_alloc(hst_v, float, pic_number * pic_number * sizeof(float));

	//Allocating device memory.
	cuda_exec(cudaMalloc(&dev_target, pic_dim * sizeof(float)));
	cuda_exec(cudaMalloc(&dev_mean, pic_dim * sizeof(float)));
	cuda_exec(cudaMalloc(&dev_matrix, pic_dim * pic_number * sizeof(float)));
	cuda_exec(cudaMalloc(&dev_u, pic_dim * pic_dim * sizeof(float)));
	cuda_exec(cudaMalloc(&dev_result_matrix, pic_dim * pic_number * sizeof(float)));
	cuda_exec(cudaMalloc(&dev_result_target, pic_dim * sizeof(float)));
	cuda_exec(cudaMalloc(&dev_norms, pic_number * sizeof(float)));

	cuda_exec(cudaMemset(dev_mean, 0, pic_dim * sizeof(float)));

	//Initializing of handle for cublas.
	cublas_exec(cublasCreate(&handle));
	cublas_exec(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));


	//Loading pictures in device memory and calculating mean value per pixel.
	load_pictures(argv[2], pic_number, pic_dim, hst_matrix, dev_matrix, dev_mean, handle, target.data(), dev_target);

	//Normalization of pictures data values.
	//	matrix = matrix - mean value per each element.
	//	vector = vector - mean value per each element.
	float alfa = -1;
	for(int i = 0; i < pic_number; i++)
		cublas_exec(cublasSaxpy(handle, pic_dim, &alfa, dev_mean, 1, i*pic_dim + dev_matrix, 1 ));

	cublas_exec(cublasSaxpy(handle, pic_dim, &alfa, dev_mean, 1, dev_target, 1 ));


	//Returning of newly calculated matrix in host memory (needed for magma svd calculating).
	cuda_exec(cudaMemcpy(hst_matrix, dev_matrix, pic_dim * pic_number * sizeof(float), cudaMemcpyDeviceToHost));

	//Calculating unitary matrix U from SVD.
	calculating_svd(pic_dim, pic_number, hst_matrix, hst_sing_values, hst_u, hst_v);
	//Copying matrix U from host to device because magma returns values on host..
	cuda_exec(cudaMemcpy(dev_u, hst_u, pic_dim * pic_dim * sizeof(float), cudaMemcpyHostToDevice));


	//y = U^T * matrix
	//Eigenface representation for mean matrix and target picture.
	alfa = 1;
	float beta = 0;
	cublas_exec(cublasSgemm(handle,  CUBLAS_OP_T,  CUBLAS_OP_N, pic_dim, pic_number, pic_dim, &alfa, dev_u, pic_dim, dev_matrix, pic_dim, &beta, dev_result_matrix, pic_dim));
	cublas_exec(cublasSgemm(handle,  CUBLAS_OP_T,  CUBLAS_OP_N, pic_dim, 1, pic_dim, &alfa, dev_u, pic_dim, dev_target, pic_dim, &beta, dev_result_target, pic_dim));


	//Subtraction target picture from training set data for looking the closest one.
	//Finding closest one by looking for norm2 on each row.
	//Dividing each picture from training set with our target picture so we can find
	//the most similar one.
	//Each row norm is saved on host memory (better on device if it's possible).
	alfa = -1;
	float res;
	//Maybe could go in parallel.
	for(int i = 0; i < pic_number; i++) {
		cublas_exec(cublasSaxpy(handle, pic_dim, &alfa, dev_result_target, 1, i*pic_dim + dev_result_matrix, 1 ));
        cublas_exec(cublasSnrm2(handle, pic_dim, i*pic_dim + dev_result_matrix, 1, hst_norms+i));
	}

	//Searching for exact column that is closest by norm2 to our target picture.
	int result;
	cuda_exec(cudaMemcpy(dev_norms, hst_norms, pic_number * sizeof(float), cudaMemcpyHostToDevice));
	cublasIsamin(handle, pic_number, dev_norms, 1, &result);
	result--;

	printf("Target picture is in directory s%d, and closest match is picture %d.pgm\n", result/10+1, result%10+1);

//Deallocating memory
end:
	free(hst_matrix);
	free(hst_norms);

	free(hst_sing_values);
	free(hst_u);
	free(hst_v);

	cuda_exec(cudaFree(dev_target));
	cuda_exec(cudaFree(dev_mean));
	cuda_exec(cudaFree(dev_matrix));
	cuda_exec(cudaFree(dev_u));
	cuda_exec(cudaFree(dev_result_matrix));
	cuda_exec(cudaFree(dev_result_target));
	cuda_exec(cudaFree(dev_norms));

	cublas_exec(cublasDestroy(handle));

	return 0;
}
