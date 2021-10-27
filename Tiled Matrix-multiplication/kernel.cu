/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#define TILE_SIZE  16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE
    

    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];    

    float Ctot = 0.0;
    for(int i = 0; i < (n-1)/TILE_SIZE+1; i++){
       if(Row < m && i*TILE_SIZE+tx < n){
       sA[ty][tx] = A[Row*n + i*TILE_SIZE+tx];
       }else{
       sA[ty][tx] = 0.0;
       }
       
       if(i*TILE_SIZE+ty < n && Col < k){
       sB[ty][tx] = B[(i*TILE_SIZE+ty)*k + Col];
       }else{
       sB[ty][tx] = 0.0;
       }
       __syncthreads();
        for(int j = 0; j < TILE_SIZE; j++){
           Ctot += sA[ty][j] * sB[j][tx];
           }
       __syncthreads();
      } 
if(Row < m && Col < k){
 C[Row * k + Col] = Ctot;
}
      
    




}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;

    //INSERT CODE HERE
    dim3 gridDim((n-1)/BLOCK_SIZE + 1, (m-1)/BLOCK_SIZE + 1, 1);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);



    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE

    mysgemm<<< gridDim, blockDim >>> (m, n, k, A, B, C);


}


