#ifndef __FRACTAL_HOST_CODE__
#define __FRACTAL_HOST_CODE__

// Includes
#include "DeviceCode.cuh"
#include "lodepng.h"
#include <cuda_runtime.h>
#include <boost\program_options.hpp>
#include <boost\property_tree\ptree.hpp>
#include <boost\property_tree\xml_parser.hpp>
#include <boost\foreach.hpp>
#include <exception>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>
#include <cmath>

// Preset file
#define PRESET_FILE "./presets.xml"

// Program options
namespace po = boost::program_options;
namespace pt = boost::property_tree;

/**
 *
 *
 *
 *
 */
unsigned parseHexValue(std::string hexstring) {
	unsigned value;
	std::istringstream iss(hexstring);
	iss >> std::hex >> value;
	return value;
}

/**
 *
 *
 *
 *
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
 *
 *
 *
 *
 */
fColor parseFColorFromTree(pt::ptree tree) {
	return fColor(
		tree.get<float>("<xmlattr>.r"),
		tree.get<float>("<xmlattr>.g"),
		tree.get<float>("<xmlattr>.b"));
};

/**
 *
 *
 *
 *
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
 *
 *
 *
 *
 */
colormap parseColormapFromPreset(std::string name) {
	// Default colormap
	colormap cmap;

	// Filestream
	std::ifstream preset_file(PRESET_FILE);
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
	float consr, consi;
	unsigned width, height;
	std::string cname, fname;

	// Get user input
	po::options_description options("> CUDAFractal [options]");
	options.add_options()
		("help", "print help message")
		("cr", po::value<float>(&consr)->default_value(-0.4), "real value of c")
		("ci", po::value<float>(&consi)->default_value(0.6), "imaginary value of c")
		("width", po::value<unsigned>(&width)->default_value(1920), "image width")
		("height", po::value<unsigned>(&height)->default_value(1080), "image height")
		("cmap", po::value<std::string>(&cname)->default_value("blackwhite"), "colormap preset")
		("file", po::value<std::string>(&fname), "output file name");
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
	colormap cmap;
	try {
		cmap = parseColormapFromPreset(cname);
	}
	catch (std::exception& err) {
		std::cout << "ERROR (parsing colormap): ";
		std::cout << err.what() << std::endl;
		std::cout << "\"q\" to exit...";
		char q; std::cin >> q;
		return 1;
	}

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