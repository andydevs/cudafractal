// Includes
#include "Generate.h"
#include "ColorKernels.cuh"
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

	// Create a fractal space buffer
	byte* space;
	cudaMallocManaged(&space, width*height*sizeof(byte));

	// Call Fractal Kernel
	DOING("Running fractal kernel for " + mnemonic);
	if (mbrot) { mandelbrotset_launcher(scale, trans, width, height, space); }
	else { juliaset_launcher(cons, scale, trans, width, height, space); }
	DONE();

	// Create an image buffer as cuda unified memory and save location at image
	byte* image;
	cudaMallocManaged(&image, width*height*IMAGE_NUM_CHANNELS*sizeof(byte));

	// Call Colormap Kernel
	DOING("Running colormap kernel for " + mnemonic);
	legacy_colormap_launcher(cmap, width, height, space, image);
	DONE();

	// Save img buffer to png file
	DOING("Saving png");
	lodepng_encode32_file(filename.c_str(), image, width, height);
	DONE();

	// Free buffers
	cudaFree(space);
	cudaFree(image);
};