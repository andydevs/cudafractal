// Includes
#include "DeviceCode.cuh"

/**
 * Returns complex from the given pixel in the image
 *
 * @param x the x value of the pixel
 * @param y the y value of the pixel
 * @param w the width of the image
 * @param h the height of the image
 *
 * @return complex from the given pixel in the image
 */
static __device__ __host__ __inline__
cuFloatComplex fromPixel(unsigned x, unsigned y, unsigned w, unsigned h, cuFloatComplex s, cuFloatComplex t) {
	cuFloatComplex z = make_cuFloatComplex(
		((float)2.0) * ((float)w/h) * x/w - ((float)w/h),
		((float)2.0) * y/h - ((float)1.0));

	// Return transform
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
static __device__ __host__ __inline__
unsigned char iterations(cuFloatComplex w, cuFloatComplex c) {
	// Set initial z value
	cuFloatComplex z = w;

	// Algorithm
	unsigned char iters;
	for (iters = 0; iters < 255; iters++) {
		// Break if abs(z) >= 2
		if (cuCabsf(z) >= 2) break;

		// Run iteration of z: z = z^2 + c
		z = cuCaddf(cuCmulf(z, z), c);
	}

	// Return iterations
	return iters;
}

/**
 * It assigns the corresponding pixel of the thread to a corresponding starting
 * complex number z. Then, it runs the juliaset algorithm on z using the given c.
 * Finally, it computes the color from the resulting iteration number and assigns 
 * that color to the thread's corresponding pixel in the image.
 *
 * @param c    the complex constant c
 * @param cmap the colormap to use when mapping colors
 * @param w    the width of the image
 * @param h    the height of the image
 * @param img  the image buffer
 */
__global__
void juliaset(cuFloatComplex c, cuFloatComplex s, cuFloatComplex t, colormap cmap, unsigned w, unsigned h, byte* img) {
	// Get x and y of image (don't run pixels beyond size on img)
	unsigned y = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned x = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;

	// Run iterations algorithm, setting w to the pixel complex, then set pixel in image to mapped color
	setPixel(img, w, h, x, y,
		mapColor(cmap, 
			iterations(
				fromPixel(x, y, w, h, s, t), c
	)));
};

/**
* It assigns the corresponding pixel of the thread to a corresponding complex
* constant number c and sets z to 0. Then, it runs the iteration algorithm on
* z using the given c.Finally, it computes the color from the resulting iteration
* number and assigns that color to the thread's corresponding pixel in the image.
*
* @param cmap the colormap to use when mapping colors
* @param w    the width of the image
* @param h    the height of the image
* @param img  the image buffer
*/
__global__
void mandelbrotset(cuFloatComplex s, cuFloatComplex t, colormap cmap, unsigned w, unsigned h, unsigned char* img) {
	// Get x and y of image (don't run pixels beyond size on img)
	unsigned y = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned x = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;

	// Run iterations algorithm, setting w to 0 and c to the pixel complex, 
	// then set pixel in image to mapped color
	setPixel(img, w, h, x, y,
		mapColor(cmap,
			iterations(
				make_cuFloatComplex(0.0, 0.0),
				fromPixel(x, y, w, h, s, t)
	)));
}