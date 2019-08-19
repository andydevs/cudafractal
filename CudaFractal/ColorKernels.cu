#include "Coloring.cuh"
#include "ColorKernels.cuh"

// Cuda stuff
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

/**
 * Use legacy colormap code to map number from iteration space to byte color in image
 *
 * @param cmap  colormap struct used
 * @param w     width of image
 * @param h     height of image
 * @param space fractal iteration space buffer
 * @param image final image buffer
 */
__global__
void legacy_colormap_kernel(colormap cmap, unsigned w, unsigned h, byte *space, byte *image) {
	// Get x and y of image (don't run pixels beyond size on img)
	unsigned y = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned x = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;

	// Get iters
	byte iter = space[y*w + x];
	color col = mapColor(cmap, iter);
	setPixel(image, w, h, x, y, col);
}

/**
 * Launch legacy colormap kernel
 *
 * Use legacy colormap code to map number from iteration space to byte color in image
 *
 * @param cmap  colormap struct used
 * @param w     width of image
 * @param h     height of image
 * @param space fractal iteration space buffer
 * @param image final image buffer
 */
void legacy_colormap_launcher(colormap cmap, unsigned w, unsigned h, byte *space, byte *image) {
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
	legacy_colormap_kernel<<<gridSpace, blockSpace>>>(cmap, w, h, space, image);
	cudaDeviceSynchronize(); // Wait for kernel to finish
}