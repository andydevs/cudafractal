#include "lodepng.h"
#include "cuda_runtime.h"
#include <stdlib.h>
#include <stdio.h>

/**
 * Runs an operation on the given pixel space
 *
 * @param img_w the width of the image
 * @param img_h the height of the image
 * @param img_c the color channels of the image
 * @param img   the image buffer
 */
__global__
void pixels(unsigned img_w, unsigned img_h, unsigned img_c, unsigned char* img) {
	// Get x and y of image
	// Don't run pixels beyond size on cu_img
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	int x = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= img_w || y >= img_h) return;
	
	// Get colors
	unsigned char r = (x * 256) / img_w;
	unsigned char g = (x * 256) / img_w;
	unsigned char b = (y * 256) / img_h;
	unsigned char a = 0xff;

	// Append colors to image buffer
	img[(y*img_w + x)*img_c + 0] = r;
	img[(y*img_w + x)*img_c + 1] = g;
	img[(y*img_w + x)*img_c + 2] = b;
	img[(y*img_w + x)*img_c + 3] = a;
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
	unsigned img_c = 4;
	unsigned img_l = img_h*img_w*img_c;
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
	pixels<<<gridSize, blockSize>>>(img_w, img_h, img_c, img);
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