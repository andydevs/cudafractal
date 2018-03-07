#ifndef __FRACTAL_HOST_CODE__
#define __FRACTAL_HOST_CODE__

// Includes
#include "DeviceCode.cuh"
#include "lodepng.h"
#include <cuda_runtime.h>
#include <boost\program_options.hpp>

// Libraries
#include <exception>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <map>

// Start time
clock_t start;

// Macros
#define DOING(task) \
	std::cout << task << "..."; \
	start = clock();
#define DONE() \
	std::cout << "Done! " \
		<< (float)(clock() - start) / CLOCKS_PER_SEC << "s" \
		<< std::endl;

// Boost namespaces
namespace po = boost::program_options;

/**
 * Returns the preset colormap of the given name
 *
 * @param name the name of the colormap
 * 
 * @return the preset colormap
 */
colormap fromPreset(std::string name) {
	// Presets map
	std::map<std::string, colormap> presets;

	// Populate presets map
	presets["blackwhite"] = colormap::gradient(
		color::hex(0x000000), 
		color::hex(0xffffff));
	presets["nvidia"] = colormap::gradient(
		color::hex(0x000000),
		color::hex(0xa3ff00));
	presets["saffron"] = colormap::sinusoid(
		fColor(1.4, 1.4, 1.4),
		fColor(-2.0, -3.0, -4.0),
		0xff);

	// Return appropriate preset
	return presets[name];
};


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
	bool help, mbrot;
	float consr, consi;
	unsigned width, height;
	std::string cname, fname;

	// Get user input
	po::options_description options("> CUDAFractal [options]");
	options.add_options()
		("help", po::bool_switch(&help), "print help message")
		("mbrot", po::bool_switch(&mbrot), "compute the mandelbrot fractal algorithm")
		("cr", po::value<float>(&consr)->default_value(-0.4), "real value of c")
		("ci", po::value<float>(&consi)->default_value(0.6), "imaginary value of c")
		("width", po::value<unsigned>(&width)->default_value(1920), "image width")
		("height", po::value<unsigned>(&height)->default_value(1080), "image height")
		("cmap", po::value<std::string>(&cname)->default_value("nvidia"), "colormap preset")
		("file", po::value<std::string>(&fname), "output file name");
	po::variables_map vars;
	po::store(po::parse_command_line(argc, argv, options), vars);
	po::notify(vars);

	// Exit if no filename specified!
	if (fname.empty()) {
		std::cout << "ERROR: No filename specified!" << std::endl;
		return 1;
	}

	// Get colormap and constant
	colormap cmap = fromPreset(cname);
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
	// Each thread computes a separate pixel in the Julia/mandelbrot set
	if (mbrot) {
		DOING("Running Mandelbrot set kernel");
		mandelbrotset<<<gridSpace, blockSpace>>>(cmap, width, height, image);
		cudaDeviceSynchronize(); // Wait for kernel to finish
		DONE();
	}
	else {
		DOING("Running Julia set kernel");
		juliaset<<<gridSpace, blockSpace>>>(cons, cmap, width, height, image);
		cudaDeviceSynchronize(); // Wait for kernel to finish
		DONE();
	}

	// Save img buffer to png file
	DOING("Saving png");
	lodepng_encode32_file(fname.c_str(), image, width, height);
	DONE();
	
	// Free image buffer and exit
	cudaFree(image);
	return 0;
}

#endif // !__FRACTAL_HOST_CODE__