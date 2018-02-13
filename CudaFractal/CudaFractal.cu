#include "lodepng.h"
#include "cuda_runtime.h"
#include "cuComplex.h"
#include <iostream>

// PNG Image format
#define IMAGE_NUM_CHANNELS 4
#define IMAGE_RED_CHANNEL 0
#define IMAGE_GREEN_CHANNEL 1
#define IMAGE_BLUE_CHANNEL 2
#define IMAGE_ALPHA_CHANNEL 3

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
		-2.0 * ((float)w / h) * x / w + ((float)w / h),
		2.0 * y / h - 1.0);
}

/**
 * The iterative process in the julia set. Computes z = z^2 + c 
 * iteratively, with z being initialized to w. Returns the number 
 * of iterations before abs(z) >= 2 (max 255).
 *
 * @param wr real part of w complex
 * @param wi imaginary part of w complex
 * @param cr real part of c complex
 * @param ci imaginary part of c complex
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
 * @param img_w the width of the image
 * @param img_h the height of the image
 * @param img   the image buffer
 */
__global__
void juliaset(cuFloatComplex c, unsigned w, unsigned h, unsigned char* img) {
	// Get x and y of image (don't run pixels beyond size on img)
	unsigned y = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned x = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;

	// Run iterations algorithm, setting w to the pixel complex
	char iters = iterations(fromPixel(x, y, w, h), c);

	// Append colors to image buffer
	img[(y*w + x)*IMAGE_NUM_CHANNELS + IMAGE_RED_CHANNEL]   = iters; // Red
	img[(y*w + x)*IMAGE_NUM_CHANNELS + IMAGE_GREEN_CHANNEL] = iters; // Green
	img[(y*w + x)*IMAGE_NUM_CHANNELS + IMAGE_BLUE_CHANNEL]  = iters; // Blue
	img[(y*w + x)*IMAGE_NUM_CHANNELS + IMAGE_ALPHA_CHANNEL] = 0xff;  // Alpha
}

/**
 * The main procedure
 *
 * @param argc the number of command line args
 * @param argv the command line args
 *
 * @return status code
 */
int main(int argc, const char* argv[]) {
	// Soon-to-be user-inputted data
	float consr = -0.4;
	float consi = 0.6;
	unsigned width = 1920;
	unsigned height = 1080;
	const char* filename = "C:\\Users\\akans\\Desktop\\fractal.png";

	// Create constant
	cuFloatComplex cons = make_cuFloatComplex(consr, consi);

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
	int gridSize = (width >= height ? width : height) / blockSize;
	dim3 gridSpace(gridSize, gridSize);

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
	// Each thread computes a separate pixel in the JuliaSet
	std::cout << "Running JuliaSet kernel...";
	juliaset<<<gridSpace, blockSpace>>>(cons, width, height, image);
	cudaDeviceSynchronize(); // Wait for kernel to finish
	std::cout << "Done!" << std::endl;

	// Save img buffer to png file
	std::cout << "Saving png...";
	lodepng_encode32_file(filename, image, width, height);
	std::cout << "Done!" << std::endl;
	
	// Free image buffer and exit
	cudaFree(image);
	return 0;
}