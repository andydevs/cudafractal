// Includes
#include "Super.h"
#include "DeviceCode.cuh"

// Maximum iterations
#define MAX_ITER 255

/**
 * Returns complex from the given pixel in the image
 *
 * @param x the x value of the pixel
 * @param y the y value of the pixel
 * @param w the width of the image
 * @param h the height of the image
 * @param s the scale complex
 * @param t the translation complex
 *
 * @return complex from the given pixel in the image
 */
__device__
cuFloatComplex fromPixel(unsigned x, unsigned y, unsigned w, unsigned h, cuFloatComplex s, cuFloatComplex t) {
	// z complex value from image pixel
	cuFloatComplex z = make_cuFloatComplex(
		((float)2.0) * ((float)w/h) * x/w - ((float)w/h),
		((float)2.0) * y/h - ((float)1.0));

	// Transform complex value
	return cuCmulf(s, cuCaddf(t, z));
}

/**
 * The iterative process in the julia set. Computes z = z^2 + c
 * iteratively, with z being initialized to w. Returns the number
 * of iterations before abs(z) >= 2 (max 255).
 *
 * @param w complex value w
 * @param c complex value c
 *
 * @return number of iterations before abs(z) >= 2 (max 255).
 */
__device__ 
unsigned char iterations(cuFloatComplex w, cuFloatComplex c) {
	// Set initial z value
	cuFloatComplex z = w;

	// Algorithm
	byte iters;
	for (iters = 0; iters < MAX_ITER; iters++) {
		// Break if abs(z) >= 2
		if (cuCabsf(z) >= 2) break;

		// Run iteration of z: z = z^2 + c
		z = cuCaddf(cuCmulf(z, z), c);
	}

	// Return iterations
	return iters;
}