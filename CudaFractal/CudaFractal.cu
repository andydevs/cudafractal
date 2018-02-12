#include "lodepng.h"
#include "cuda_runtime.h"
#include <stdlib.h>
#include <stdio.h>

// PNG Image format
#define IMAGE_NUM_CHANNELS 4
#define IMAGE_RED_CHANNEL 0
#define IMAGE_GREEN_CHANNEL 1
#define IMAGE_BLUE_CHANNEL 2
#define IMAGE_ALPHA_CHANNEL 3

/**
 * Runs an operation on the given pixel space
 *
 * @param img_w the width of the image
 * @param img_h the height of the image
 * @param img_c the color channels of the image
 * @param img   the image buffer
 */
__global__
void juliaset(unsigned img_w, unsigned img_h, unsigned char* img) {
	// Get x and y of image (don't run pixels beyond size on img)
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	int x = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= img_w || y >= img_h) return;

	// Get real and imaginary components of constant
	float cr = -0.4;
	float ci = 0.6;
	
	// Get real and imaginary components of starting complex
	float zr = -2.0 * x / img_w + 1.0;
	float zi =  2.0 * y / img_h - 1.0;

	/**
	 * Since the complex values are being calculated manually (for now),
	 * a copy of the last real value must be kept in order to properly
	 * calculate both the next imaginary value. Since the imaginary value
	 * is calculated after the next real is calculated, using the same
	 * buffer would mean using the new calculated real as part of the 
	 * calculation for the new imaginary value, which would produce 
	 * the wrong imaginary value for the iteration.
	 */
	float lr;

	// Juliaset Algorithm
	int iters;
	for (iters = 0; iters < 255; iters++) {
		// Break if abs(z)**2 >= 4
		if (zr*zr + zi*zi >= 4) break;

		// Run iteration of z: z = z^2 + c
		lr = zr;
		zr = lr*lr - zi*zi + cr;
	 	zi = 2*lr*zi + ci;
	}

	// Append colors to image buffer
	img[(y*img_w + x)*IMAGE_NUM_CHANNELS + IMAGE_RED_CHANNEL]   = iters; // Red
	img[(y*img_w + x)*IMAGE_NUM_CHANNELS + IMAGE_GREEN_CHANNEL] = iters; // Green
	img[(y*img_w + x)*IMAGE_NUM_CHANNELS + IMAGE_BLUE_CHANNEL]  = iters; // Blue
	img[(y*img_w + x)*IMAGE_NUM_CHANNELS + IMAGE_ALPHA_CHANNEL] = 0xff;  // Alpha
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
	// Image meta
	unsigned img_w = 1920;
	unsigned img_h = 1080;
	unsigned img_l = img_h*img_w*IMAGE_NUM_CHANNELS;
	unsigned char* img;

	// Process meta
	int block_n = 8;
	int grid_n = (img_w > img_h ? img_w : img_h) / block_n;
	dim3 blockSize(block_n, block_n);
	dim3 gridSize(grid_n, grid_n);

	// Cuda image buffer
	cudaMallocManaged(&img, sizeof(unsigned char)*img_l);

	// Call Cuda Routine
	printf("Running CUDA routine...");
	juliaset<<<gridSize, blockSize>>>(img_w, img_h, img);
	cudaDeviceSynchronize();
	printf("Done!\n");

	// Save to png file
	printf("Saving png...");
	lodepng_encode32_file(
		"C:\\Users\\akans\\Desktop\\fractal.png", 
		(const unsigned char*)img, img_w, img_h);
	printf("Done!\n");
	
	// Free resources and exit
	cudaFree(img);
	return 0;
}