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
 * Prints the image array to the console
 *
 * @param img_w the width of the image
 * @param img_h the height of the image
 * @param img_c the number of color channels of the image
 * @param img   the image buffer
 */
void printArray(unsigned img_w, unsigned img_h, unsigned img_c, unsigned char* img) {
	for (unsigned y = 0; y < img_h; y++) {
		for (unsigned x = 0; x < img_w; x++) {
			for (unsigned z = 0; z < img_c; z++) {
				printf("%02x", img[(y*img_w + x)*img_c + z]);
			}
			printf(" ");
		}
		printf("\n");
	}
	printf("\n");
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
	unsigned img_w = 16;
	unsigned img_h = 16;
	unsigned img_c = 4;
	unsigned img_l = img_h*img_w*img_c;
	unsigned char* img;

	// Process meta
	int block_w = 8;
	int block_h = 8;
	dim3 blockSize(block_w, block_h);
	dim3 gridSize(img_w/block_w, img_h/block_h);

	// Cuda image buffer
	cudaMallocManaged(&img, sizeof(unsigned char)*img_l);

	// Print initial
	printf("Initial:\n");
	printArray(img_w, img_h, img_c, img);

	// Call Cuda Routine
	pixels<<<gridSize, blockSize>>>(img_w, img_h, img_c, img);
	cudaDeviceSynchronize();

	// Print final
	printf("Final:\n");
	printArray(img_w, img_h, img_c, img);
	
	// Free resources
	cudaFree(img);

	// Q to quit
	printf("\"q\" to quit...");
	getchar();
	return 0;
}