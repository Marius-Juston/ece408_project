#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


#define TILE_WIDTH 16

__constant__ float Mc[3136];

__global__ void conv_forward_kernel(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) Mc[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int b = blockIdx.z;

    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    __shared__ float rowShared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float colShared[TILE_WIDTH][TILE_WIDTH];

    const unsigned int numBlocks = ceil(C * K * K / (float) TILE_WIDTH);

    const unsigned int W_BASE = C * K * K;

    float sum = 0.0;

    
    for (int i = 0; i < numBlocks; ++i){
        int tileCol = i * TILE_WIDTH + tx; // For the kernel
        int tileRow = i * TILE_WIDTH + ty; // for the input

        // input matrix shared memeory

        int X_c = tileRow / (K * K);

        int temp = tileRow % (K * K) ;
        int X_p =  temp / K;
        int X_q = temp % K;
        
        int X_h = col / W_out;
        int X_w = col % W_out;

        int K_m = row;
        int K_c = tileCol / (K * K);
        int K_h = (tileCol % (K * K)) / K;
        int K_w = (tileCol % (K * K)) % K;

        if(tileRow < W_BASE && col < H_out * W_out){
            colShared[ty][tx] = x4d(b, X_c, X_h + X_p , X_w + X_q);
        }else{
            colShared[ty][tx] = 0.0f;            
        }

        if(tileCol < W_BASE && K_m < M){
            rowShared[ty][tx] = k4d(K_m , K_c, K_h, K_w);
        }else{
            rowShared[ty][tx] = 0.0f;
        }

        __syncthreads();

        for(int k = 0; k < TILE_WIDTH; ++k){
            sum += colShared[k][tx] * rowShared[ty][k];            
        }

        __syncthreads();
    }

    int new_b = b;
    int new_m = row;
    int new_h = col / W_out;
    int new_w = col % W_out;

    if(new_m < M && col < W_out * H_out){
        y4d(new_b, new_m, new_h, new_w) = sum;
    }


#undef y4d
#undef x4d
#undef k4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const unsigned int sizeX = B * C * H * W * sizeof(float);
    const unsigned int sizeY = B * M * H_out * W_out * sizeof(float);
    const unsigned int sizeK = M * C * K * K * sizeof(float);

    cudaMalloc((void **)device_x_ptr, sizeX);
    cudaMalloc((void **)device_y_ptr, sizeY);

    cudaMemcpy(*device_x_ptr, host_x, sizeX, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(Mc, host_k, sizeK, 0 , cudaMemcpyHostToDevice);

    std::cout << sizeK << " " << C << " " << K << std::endl;
}


__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel
    const unsigned int H_out = H - K + 1;
    const unsigned int W_out = W - K + 1;

    const unsigned int M_ = ceil(M / (float) TILE_WIDTH);

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim( ceil((W_out * H_out) / (float) ( TILE_WIDTH)), M_, B);

    conv_forward_kernel<<<gridDim,  blockDim >>>(device_y, device_x,  B, M , C, H, W, K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // Copy the output back to host
    const unsigned int sizeY = B * M * H_out * W_out * sizeof(float);
    cudaMemcpy(host_y, device_y, sizeY, cudaMemcpyDeviceToHost);

    // Free device memory

    cudaFree(device_x);
    cudaFree(device_y);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
