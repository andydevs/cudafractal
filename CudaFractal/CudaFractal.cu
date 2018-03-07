#ifndef __FRACTAL_HOST_CODE__
#define __FRACTAL_HOST_CODE__

// Includes
#include "DeviceCode.cuh"
#include "lodepng.h"
#include <cuda_runtime.h>

// Bosts
#include <boost\program_options.hpp>
#include <boost\property_tree\ptree.hpp>
#include <boost\property_tree\xml_parser.hpp>
#include <boost\foreach.hpp>
#include <boost\filesystem.hpp>

// Libraries
#include <exception>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>
#include <cmath>
#include <ctime>

// Start time
clock_t start;

// Macros
#define DOING(task) \
	std::cout << task << "..."; \
	start = clock();
#define DONE() \
	std::cout << "Done! " \
		<< (float)(clock() - start) / CLOCKS_PER_SEC \
		<< "s" << std::endl;

// Boost namespaces
namespace po = boost::program_options;
namespace pt = boost::property_tree;
namespace fs = boost::filesystem;

// Preset file
#define PRESET_FILE "presets.xml"

/**
 * Returns the location of the preset file given the
 * location of the program
 *
 * @param argpath the location of the program
 *
 * @return the location of the preset file
 */
std::string getPresetFileLocation(const char* progpath) {
	return fs::system_complete(fs::path(progpath))
		.parent_path()
		.append<std::string>(PRESET_FILE)
		.string();
};

/**
 * Parses the hex value from the given string
 *
 * @param hexstring the string to parse
 *
 * @return the hex value
 */
unsigned parseHexValue(std::string hexstring) {
	unsigned value;
	std::istringstream iss(hexstring);
	iss >> std::hex >> value;
	return value;
};

/**
 * Parses a color from the given tree
 *
 * @param tree the property tree to parse
 *
 * @return the color parsed from tree
 */
color parseColorFromTree(pt::ptree tree) {
	if (tree.get<std::string>("type") == "hex") {
		return color::hex(parseHexValue(tree.get<std::string>("value")));
	}
	else if (tree.get<std::string>("type") == "hexa") {
		return color::hexa(parseHexValue(tree.get<std::string>("value")));
	}
	else if (tree.get<std::string>("type") == "rgb") {
		return color(
			tree.get<byte>("value.<xmlattr>.r"),
			tree.get<byte>("value.<xmlattr>.g"),
			tree.get<byte>("value.<xmlattr>.b"));
	}
	else if (tree.get<std::string>("type") == "rgba") {
		return color(
			tree.get<byte>("value.<xmlattr>.r"),
			tree.get<byte>("value.<xmlattr>.g"),
			tree.get<byte>("value.<xmlattr>.b"),
			tree.get<byte>("value.<xmlattr>.a"));
	}
	else return color();
};

/**
 * Parses a float color from the given property tree
 *
 * @param tree the property tree to parse
 *
 * @return the float color parsed from tree
 */
fColor parseFColorFromTree(pt::ptree tree) {
	return fColor(
		tree.get<float>("<xmlattr>.r"),
		tree.get<float>("<xmlattr>.g"),
		tree.get<float>("<xmlattr>.b"));
};

/**
 * Parses colormap from the given property tree
 *
 * @param tree the property tree to parse
 *
 * @return the colormap parsed from the tree
 */
colormap parseColormapFromTree(pt::ptree tree) {
	if (tree.get<std::string>("type") == "gradient")
		return colormap::gradient(
			parseColorFromTree(tree.get_child("from")),
			parseColorFromTree(tree.get_child("to")));
	else if (tree.get<std::string>("type") == "sinusoid")
		return colormap::sinusoid(
			parseFColorFromTree(tree.get_child("frequency")),
			parseFColorFromTree(tree.get_child("phase")),
			tree.get("alpha", 0xff));
	else return colormap();
};

/**
 * Parses colormap from the preset with the given name
 *
 * @param name the name of the colormap preset
 * @param progpath the path of the program file
 *
 * @return the colormap preset
 */
colormap parseColormapFromPreset(std::string name, const char* progpath) {
	// Default colormap
	colormap cmap;

	// Filestream
	std::ifstream preset_file(getPresetFileLocation(progpath));
	try {

		// Get ptree
		pt::ptree presets;
		read_xml(preset_file, presets);

		// Get cmap from presets
		pt::ptree preset;
		BOOST_FOREACH(pt::ptree::value_type entry, presets.get_child("presets")) {
			if (entry.second.get<std::string>("name") == name) {
				cmap = parseColormapFromTree(entry.second);
			}
		}

	}
	catch (std::exception& err) {
		preset_file.close();
		throw err;
	}

	// Close and return
	preset_file.close();
	return cmap;
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

	// Get colormap
	DOING("Parsing colormap")
	colormap cmap;
	try {
		cmap = parseColormapFromPreset(cname, argv[0]);
	}
	catch (std::exception& err) {
		std::cout << "ERROR (parsing colormap): ";
		std::cout << err.what() << std::endl;
	}
	DONE();

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