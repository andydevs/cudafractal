// Includes
#include "Generate.cuh"
#include "FractalKernels.cuh"
#include "lodepng.h"

// CUDA
#include <cuda_runtime.h>
#include <cuComplex.h>

// Libraries
#include <iostream>
#include <string>
#include <ctime>

/**
 * Launch juliaset kernel
 *
 * It assigns the corresponding pixel of the thread to a corresponding starting
 * complex number z. Then, it runs the juliaset algorithm on z using the given c.
 * Finally, it computes the color from the resulting iteration number and assigns
 * that color to the thread's corresponding pixel in the image.
 *
 * @param c    the complex constant c
 * @param s    the scale complex
 * @param t    the translation complex
 * @param cmap the colormap to use when mapping colors
 * @param w    the width of the image
 * @param h    the height of the image
 * @param img  the image buffer
 */
void juliaset_launcher(cuFloatComplex c, cuFloatComplex s, cuFloatComplex t, colormap cmap, unsigned w, unsigned h, unsigned char* img) {
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

	// Launch juliaset kernel
	juliaset_kernel<<<gridSpace, blockSpace>>>(c, s, t, cmap, w, h, img);
}

/**
 * Launch mandelbrotset kernel
 *
 * It assigns the corresponding pixel of the thread to a corresponding complex
 * constant number c and sets z to 0. Then, it runs the iteration algorithm on
 * z using the given c.Finally, it computes the color from the resulting iteration
 * number and assigns that color to the thread's corresponding pixel in the image.
 *
 * @param s the scale complex
 * @param t the translation complex
 * @param cmap the colormap to use when mapping colors
 * @param w    the width of the image
 * @param h    the height of the image
 * @param img  the image buffer
 */
void mandelbrotset_launcher(cuFloatComplex s, cuFloatComplex t, colormap cmap, unsigned w, unsigned h, unsigned char* img) {
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

	// Run kernel
	mandelbrotset_kernel<<<gridSpace, blockSpace>>>(s, t, cmap, w, h, img);
}

/**
 * Generate fractal image
 *
 * @param mbrot    true if generating mandelbrot set
 * @param cons     complex constant
 * @param scale    scale transformation complex
 * @param trans    translate transformation complex
 * @param cmap     colormap to generate with
 * @param width    width of the image
 * @param height   height of the image
 * @param filename name of file to save to
 * @param mnemonic used to identify generator job
 */
void generate(bool mbrot, cuFloatComplex cons, cuFloatComplex scale, cuFloatComplex trans, colormap cmap, unsigned width, unsigned height, std::string filename, std::string mnemonic) {
	DEFINE_TIMES

	// NOTE: 
	//	Investigate why grid spaces or block spaces 
	//	do not work in this case when made rectangular...

	// Create a cuda-managed image buffer and save location at image
	unsigned char* image;
	unsigned length = width*height*IMAGE_NUM_CHANNELS;
	cudaMallocManaged(&image, sizeof(unsigned char)*length);

	// Where the magic happens...
	// Call CUDA kernel on the given grid space of blocks
	// Each block being a block space of threads.
	// Each thread computes a separate pixel in the Julia/mandelbrot set
	DOING("Running kernel for " + mnemonic);
	if (mbrot) { mandelbrotset_launcher(scale, trans, cmap, width, height, image); }
	else { juliaset_launcher(cons, scale, trans, cmap, width, height, image); }
	cudaDeviceSynchronize(); // Wait for kernel to finish
	DONE();

	// Save img buffer to png file
	DOING("Saving png");
	lodepng_encode32_file(filename.c_str(), image, width, height);
	DONE();

	// Free image buffer and exit
	cudaFree(image);
};