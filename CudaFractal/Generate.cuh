#pragma once

// Includes
#include "Super.cuh"
#include "Coloring.cuh"

// CUDA
#include <cuda_runtime.h>
#include <cuComplex.h>

// Libraries
#include <iostream>
#include <string>

/**
 * Returns scale complex which incorporates rotation and zooming
 *
 * @param rotate the rotation value (in degrees)
 * @param zoom   the zoom value
 *
 * @return scale complex
 */
#define make_cuScaleComplex(rotate, zoom) \
	make_cuFloatComplex( \
		cos(rotate*F_PI / 180.0f) / zoom, \
		sin(rotate*F_PI / 180.0f) / zoom)

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
void generate(bool mbrot, cuFloatComplex cons, cuFloatComplex scale, cuFloatComplex trans, colormap cmap, unsigned width, unsigned height, std::string filename, std::string mnemonic);