/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

// Define your kernels in this file you may use more than one kernel if you
// need to

// INSERT KERNEL(S) HERE

__global__ void histogram_kernel(unsigned int* input, unsigned int* bins,
    unsigned int num_elements, unsigned int num_bins) {
    
    //Global thread index
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    //Create shared memory to calculate each bin on
    __shared__ unsigned int result[255];

    //Intialize each value of the shared memory array as 0
    if(threadIdx.x < num_bins){
    result[threadIdx.x] = 0;
    }
    
    if(threadIdx.x < num_bins){
    bins[threadIdx.x] = 0; 
    
    }
    __syncthreads();

    while(i < num_elements){   
         atomicAdd(&(result[input[i]]), 1);
         i += stride;
    }

    __syncthreads();
    
    if(threadIdx.x < num_bins){
       atomicAdd(&(bins[threadIdx.x]), result[threadIdx.x]);
    }
 
  }

__global__ void convert_kernel(unsigned int *bins32, uint8_t *bins8,
    unsigned int num_bins) {
    
    if(threadIdx.x < num_bins){
      bins8[threadIdx.x] = bins32[threadIdx.x];

    }
    
       

}

/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void histogram(unsigned int* input, uint8_t* bins, unsigned int num_elements,
        unsigned int num_bins) {

    

    // Create 32 bit bins
    unsigned int *bins32;
    cudaMalloc((void**)&bins32, num_bins * sizeof(unsigned int));
    cudaMemset(bins32, 0, num_bins * sizeof(unsigned int));

    // Launch histogram kernel using 32-bit bins
    dim3 dim_grid, dim_block;
    dim_block.x = 512; dim_block.y = dim_block.z = 1;
    dim_grid.x = 30; dim_grid.y = dim_grid.z = 1;
    histogram_kernel<<<dim_grid, dim_block, num_bins*sizeof(unsigned int)>>>
        (input, bins32, num_elements, num_bins);

    // Convert 32-bit bins into 8-bit bins
    dim_block.x = 512;
    dim_grid.x = (num_bins - 1)/dim_block.x + 1;
    convert_kernel<<<dim_grid, dim_block>>>(bins32, bins, num_bins);

    // Free allocated device memory
    cudaFree(bins32);

}


