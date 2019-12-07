#include "Coloring.cuh"
#include "ColorKernels.cuh"

// Cuda stuff
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

/**
 * Set pixel in image to the given color
 *
 * @param img   image data buffer
 * @param w     width of image
 * @param h     height of image
 * @param x     x position of pixel
 * @param y     y position of pixel
 * @param color rgba color to set pixel to
 */
__device__
byte setPixel(byte* img, unsigned w, unsigned h, unsigned x, unsigned y, rgba color) {
	img[ (y*w + x)*IMAGE_NUM_CHANNELS + IMAGE_RED_CHANNEL   ] = color.r; // Red
	img[ (y*w + x)*IMAGE_NUM_CHANNELS + IMAGE_GREEN_CHANNEL ] = color.g; // Green
	img[ (y*w + x)*IMAGE_NUM_CHANNELS + IMAGE_BLUE_CHANNEL  ] = color.b; // Blue
	img[ (y*w + x)*IMAGE_NUM_CHANNELS + IMAGE_ALPHA_CHANNEL ] = color.a; // Alpha
}

/**
 * Compute linear gradient function at given parameter
 *
 * @param from starting point of linear gradient
 * @param to   ending point of linear gradient
 * @param iter iteration value to compute gradient at
 *
 * @return linear gradient function at given parameter
 */
__device__
byte linearGradient(byte from, byte to, byte iter) {
	return from + ((to - from) * iter / BYTE_MAX);
}

/**
 * Map number from iteration space to byte color in image from linear gradient between two colors
 *
 * @param from  initial color in gradient
 * @param to    final color in gradient
 * @param w     width of image
 * @param h     height of image
 * @param space fractal iteration space buffer
 * @param image final image buffer
 */
__global__
void gradient_colormap_kernel(rgba from, rgba to, unsigned w, unsigned h, byte *space, byte *image) {
	// Get x and y of image (don't run pixels beyond size on img)
	unsigned y = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned x = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;

	// Calculate linear gradient
	rgba color;
	byte iter = space[y*w + x];
	color.r = linearGradient(from.r, to.r, iter);
	color.g = linearGradient(from.g, to.g, iter);
	color.b = linearGradient(from.b, to.b, iter);
	color.a = linearGradient(from.a, to.a, iter);

	// Set pixel in image
	setPixel(image, w, h, x, y, color);
}

/**
 * Launch gradient colormap kernel
 *
 * Map number from iteration space to byte color in image from linear gradient between two colors
 *
 * @param from  initial color in gradient
 * @param to    final color in gradient
 * @param w     width of image
 * @param h     height of image
 * @param space fractal iteration space buffer
 * @param image final image buffer
 */
void gradient_colormap_launcher(rgba from, rgba to, unsigned w, unsigned h, byte *space, byte *image) {
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
	gradient_colormap_kernel<<<gridSpace, blockSpace>>>(from, to, w, h, space, image);
	cudaDeviceSynchronize(); // Wait for kernel to finish
}

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