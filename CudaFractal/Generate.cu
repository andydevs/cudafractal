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