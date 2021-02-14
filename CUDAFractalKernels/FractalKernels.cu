#include "Super.h"
#include "FractalKernels.cuh"
#include "DeviceCode.cuh"
#include "Coloring.cuh"

// Cuda stuff
#include <cuComplex.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

/**
 * It assigns the corresponding pixel of the thread to a corresponding starting
 * complex number z. Then, it runs the juliaset algorithm on z using the given c
 * and saves the result to the fractal space
 *
 * @param c   the complex constant c
 * @param s   the scale complex
 * @param t   the translation complex
 * @param w   the width of the image
 * @param h   the height of the image
 * @param spc the space buffer
 */
__global__
void juliaset_kernel(cuFloatComplex c, cuFloatComplex s, cuFloatComplex t, unsigned w, unsigned h, byte* spc) {
	// Get x and y of image (don't run pixels beyond size on img)
	unsigned y = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned x = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;

	// Run iterations algorithm, and save result
	spc[y*w + x] = iterations(fromPixel(x, y, w, h, s, t), c);
};

/**
 * Launch juliaset kernel
 *
 * It assigns the corresponding pixel of the thread to a corresponding starting
 * complex number z. Then, it runs the juliaset algorithm on z using the given c
 * and saves the result to the space
 *
 * @param c   the complex constant c
 * @param s   the scale complex
 * @param t   the translation complex
 * @param w   the width of the image
 * @param h   the height of the image
 * @param spc the image buffer
 */
void juliaset_launcher(cuFloatComplex c, cuFloatComplex s, cuFloatComplex t, unsigned w, unsigned h, byte* spc) {
	// Block space
	// Using 8x8 thread block space because that 
	// divides evenly into most standard resolutions
	int blockSize = 8;
	dim3 blockSpace(blockSize, blockSize);

	// Grid space
	// Find the largest side of the image rectangle
	// and make a square out of that side. Divide 
	// number oftotal "threads" by the block size. 
	// This is the number of the blocks in the grid
	int gridSize = (w >= h ? w : h) / blockSize;
	dim3 gridSpace(gridSize, gridSize);

	// NOTE: 
	//	Investigate why grid spaces or block spaces 
	//	do not work in this case when made rectangular...

	// Where the magic happens...
	// Call CUDA kernel on the given grid space of blocks
	// Each block being a block space of threads.
	// Each thread computes a separate pixel in the julia set
	juliaset_kernel<<<gridSpace, blockSpace>>>(c, s, t, w, h, spc);
	cudaDeviceSynchronize(); // Wait for kernel to finish
};

/**
 * It assigns the corresponding pixel of the thread to a corresponding complex
 * constant number c and sets z to 0. Then, it runs the iteration algorithm on
 * z using the given c and saves the result to the space buffer
 *
 * @param s   the scale complex
 * @param t   the translation complex
 * @param w   the width of the image
 * @param h   the height of the image
 * @param spc the image buffer
 */
__global__
void mandelbrotset_kernel(cuFloatComplex s, cuFloatComplex t, unsigned w, unsigned h, byte* spc) {
	// Get x and y of image (don't run pixels beyond size on img)
	unsigned y = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned x = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;

	// Run iterations algorithm, setting w to 0 and c to the pixel complex
	spc[y*w + x] = iterations(make_cuFloatComplex(0.0, 0.0), fromPixel(x, y, w, h, s, t));	
};

/**
 * Launch mandelbrotset kernel
 *
 * It assigns the corresponding pixel of the thread to a corresponding complex
 * constant number c and sets z to 0. Then, it runs the iteration algorithm on
 * z using the given c and saves the result to the space buffer
 *
 * @param s   the scale complex
 * @param t   the translation complex
 * @param w   the width of the image
 * @param h   the height of the image
 * @param spc the image buffer
 */
void mandelbrotset_launcher(cuFloatComplex s, cuFloatComplex t, unsigned w, unsigned h, byte* spc) {
	// Block space
	// Using 8x8 thread block space because that 
	// divides evenly into most standard resolutions
	int blockSize = 8;
	dim3 blockSpace(blockSize, blockSize);

	// Grid space
	// Find the largest side of the image rectangle
	// and make a square out of that side. Divide 
	// number oftotal "threads" by the block size. 
	// This is the number of the blocks in the grid
	int gridSize = (w >= h ? w : h) / blockSize;
	dim3 gridSpace(gridSize, gridSize);

	// NOTE: 
	//	Investigate why grid spaces or block spaces 
	//	do not work in this case when made rectangular...

	// Where the magic happens...
	// Call CUDA kernel on the given grid space of blocks
	// Each block being a block space of threads.
	// Each thread computes a separate pixel in the mandelbrot set
	mandelbrotset_kernel<<<gridSpace, blockSpace>>>(s, t, w, h, spc);
	cudaDeviceSynchronize(); // Wait for kernel to finish
};