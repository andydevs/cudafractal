// Includes
#include "DeviceCode.cuh"
#include "Coloring.cuh"

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
cuFloatComplex fromPixel(unsigned x, unsigned y, unsigned w, unsigned h) {
	return make_cuFloatComplex(
		((float)-2.0) * ((float)w / h) * x / w + ((float)w / h),
		((float)2.0) * y / h - ((float)1.0));
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
 * @param c   the complex constant c
 * @param w   the width of the image
 * @param h   the height of the image
 * @param img the image buffer
 */
__global__
void juliaset(cuFloatComplex c, unsigned w, unsigned h, byte* img) {
	// Get x and y of image (don't run pixels beyond size on img)
	unsigned y = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned x = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;

	// Run iterations algorithm, setting w to the pixel complex
	byte iters = iterations(fromPixel(x, y, w, h), c);

	// Gradient color
	color col = colorGrad(iters, hex(0x000000), hex(0xa3ff00));

	// Set pixel in image
	setPixel(img, w, h, x, y, col);
}