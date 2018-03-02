#ifndef __FRACTAL_HOST_CODE__
#define __FRACTAL_HOST_CODE__

#include "DeviceCode.cuh"
#include "lodepng.h"

#include <cuda_runtime.h>
#include <boost\program_options.hpp>
#include <iostream>
#include <string>

// Program options
namespace po = boost::program_options;

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
	float consr, consi;
	unsigned width, height;
	std::string filename;

	// Get user input
	po::options_description options("> CUDAFractal [options]");
	options.add_options()
		("help", "print help message")
		("cr", po::value<float>(&consr)->default_value(-0.4), "real value of c")
		("ci", po::value<float>(&consi)->default_value(0.6), "imaginary value of c")
		("width", po::value<unsigned>(&width)->default_value(1920), "image width")
		("height", po::value<unsigned>(&height)->default_value(1080), "image height")
		("file", po::value<std::string>(&filename), "output file name");
	po::variables_map vars;
	po::store(po::parse_command_line(argc, argv, options), vars);
	po::notify(vars);

	// Exit if no filename specified!
	if (filename.empty()) {
		std::cout << "ERROR: No filename specified!" << std::endl;
		return 1;
	}

	// Create constant
	cuFloatComplex cons = make_cuFloatComplex(consr, consi);

	// Get colormap
	colormap cmap = colormap::gradient(
		color::hex(0x000000), 
		color::hex(0xa3ff00));

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
	juliaset<<<gridSpace, blockSpace>>>(cons, cmap, width, height, image);
	cudaDeviceSynchronize(); // Wait for kernel to finish
	std::cout << "Done!" << std::endl;

	// Save img buffer to png file
	std::cout << "Saving png...";
	lodepng_encode32_file(filename.c_str(), image, width, height);
	std::cout << "Done!" << std::endl;
	
	// Free image buffer and exit
	cudaFree(image);
	return 0;
}

#endif // !__FRACTAL_HOST_CODE__