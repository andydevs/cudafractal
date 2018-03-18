#ifndef __FRACTAL_HOST_CODE__
#define __FRACTAL_HOST_CODE__

// Includes
#include "DeviceCode.cuh"
#include "lodepng.h"
#include <cuda_runtime.h>

// Boost
#include <boost\foreach.hpp>
#include <boost\optional.hpp>
#include <boost\program_options.hpp>
#include <boost\property_tree\ptree.hpp>
#include <boost\property_tree\xml_parser.hpp>

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

// Global Flags
bool verbose;

// Start times
clock_t big_start;
clock_t start;

// Macros
#define BIG_DOING(task) \
	std::cout << task << std::endl \
	<< "===============================================================" << std::endl; \
	big_start = clock();
#define BIG_DONE() \
	std::cout \
		<< "===============================================================" << std::endl \
		<< "Done! " << (float)(clock() - big_start) / CLOCKS_PER_SEC << "s" << std::endl;
#define DOING(task) \
	std::cout << task << "..."; \
	start = clock();
#define DONE() \
	std::cout << "Done! " \
		<< (float)(clock() - start) / CLOCKS_PER_SEC << "s" \
		<< std::endl;
#define VERBOSE(message) \
	if (verbose) std::cout << message << std::endl;
#define COMPLEX_STRING(complex) \
	std::to_string(complex.x) + " + " + std::to_string(complex.y) + "i"

// Boost namespaces
namespace po = boost::program_options;
namespace pt = boost::property_tree;

// --------------------------------- PRESET PARSE ---------------------------------

// Presets map
bool uninitialized = true;
std::map<std::string, colormap> presets;

/**
 * Initializes the presets map
 */
void initPresets() {
	// Initialize if uninitialized
	if (uninitialized) {
		// Populate presets map
		VERBOSE("Initialize presets map");
		presets["noir"] = colormap::gradient(
			color::hex(0x000000),
			color::hex(0xffffff));
		presets["ink"] = colormap::gradient(
			color::hex(0xffffff),
			color::hex(0x000000));
		presets["nvidia"] = colormap::gradient(
			color::hex(0x000000),
			color::hex(0xa3ff00));
		presets["orchid"] = colormap::gradient(
			color::hex(0xeeeeff),
			color::hex(0xff0000));
		presets["flower"] = colormap::sinusoid(
			fColor(0.7, 0.7, 0.7),
			fColor(-2.0, -2.0, -1.0));
		presets["psychedelic"] = colormap::sinusoid(
			fColor(5.0, 5.0, 5.0),
			fColor(4.1, 4.5, 5.0));
		presets["ice"] = colormap::sinusoid(
			fColor(2.0, 2.0, 0.1),
			fColor(0.0, 0.0, 2.0));
		presets["fruity"] = colormap::sinusoid(
			fColor(5.0, 5.0, 5.0),
			fColor(0.0, 4.5, 2.5));
		presets["sarree"] = colormap::sinusoid(
			fColor(1.4, 1.4, 1.4),
			fColor(2.0, 3.0, 4.0));
		presets["saffron"] = colormap::sinusoid(
			fColor(1.00, 2.00, 2.00),
			fColor(F_P1, F_P1, F_P1));
		presets["lightgarden"] = colormap::sinusoid(
			fColor(1.00, 2.00, 9.00),
			fColor(F_N1, F_N1, F_N1));
		presets["acid"] = colormap::sinusoid(
			fColor(8.00, 9.00, 0.00),
			fColor(F_N1, F_N1, F_N1));

		// Toggle uninitialized
		uninitialized = false;
	}
};

/**
 * Returns the preset colormap of the given name
 *
 * @param name the name of the colormap
 * 
 * @return the preset colormap
 */
colormap fromPreset(std::string name) {
	initPresets();
	return presets[name];
};

/**
 * Lists all presets available
 */
void listPresets() {
	initPresets();
	std::cout << "Presets Available:" << std::endl;
	for each (std::pair<std::string, colormap> entry in presets) {
		std::cout << "    " << entry.first << std::endl;
	}
};

// -------------------------------- GENERATOR CODE --------------------------------

/**
 * Returns scale complex which incorporates rotation and zooming
 *
 * @param rotate the rotation value (in degrees)
 * @param zoom   the zoom value
 *
 * @return scale complex
 */
cuFloatComplex make_cuScaleComplex(float rotate, float zoom) {
	return make_cuFloatComplex(
		cos(rotate*F_PI / 180.0f) / zoom,
		sin(rotate*F_PI / 180.0f) / zoom);
};

/**
 * Generate fractal image
 *
 * @param mbrot    true if generating mandelbrot set
 * @param cons     complex constant
 * @param scale    scale transformation complex
 * @param trans    translate transformation complex
 * @param cmap     colormap to generate with
 * @param width    width of the image
 * @param height   height of the image
 * @param filename name of file to save to
 * @param mnemonic used to identify generator job
 */
void generate(bool mbrot, cuFloatComplex cons, cuFloatComplex scale, cuFloatComplex trans, colormap cmap, unsigned width, unsigned height, std::string filename, std::string mnemonic) {
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
	DOING("Running kernel for " + mnemonic);
	if (mbrot) { mandelbrotset<<<gridSpace, blockSpace>>>(scale, trans, cmap, width, height, image); }
	else { juliaset<<<gridSpace, blockSpace>>>(cons, scale, trans, cmap, width, height, image); }
	cudaDeviceSynchronize(); // Wait for kernel to finish
	DONE();

	// Save img buffer to png file
	DOING("Saving png");
	lodepng_encode32_file(filename.c_str(), image, width, height);
	DONE();

	// Free image buffer and exit
	cudaFree(image);
};

// ---------------------------------- XML PARSE -----------------------------------
/**
 * Parse hex value from string
 *
 * @param str the hex string
 *
 * @return hex value from string
 */
unsigned parseHex(std::string str) {
	unsigned val;
	std::stringstream stream(str);
	stream >> std::hex >> val;
	return val;
};

/**
 * Parse color from property tree
 *
 * @param col (optional) the property tree
 * 
 * @return color from property tree
 */
color parseColor(boost::optional<pt::ptree&> col) {
	if (col) {

		// Parse different types of colors
		std::string type = col->get("<xmlattr>.type", "mono");
		if (type == "hex") {
			return color::hex(parseHex(col->get("<xmlattr>.hex", "0x0000000")));
		}
		else if (type == "hexa") {
			return color::hexa(parseHex(col->get("<xmlattr>.hexa", "0x0000000")));
		}
		else if (type == "rgb") {
			return color(
				col->get("<xmlattr>.r", 0x00),
				col->get("<xmlattr>.g", 0x00),
				col->get("<xmlattr>.b", 0x00));
		}
		else if (type == "rgba") {
			return color(
				col->get("<xmlattr>.r", 0x00),
				col->get("<xmlattr>.g", 0x00),
				col->get("<xmlattr>.b", 0x00),
				col->get("<xmlattr>.a", 0x00));
		}
		else {
			return color();
		}

	}
	else {
		return color();
	}
};

/**
 * Parse float color from property tree
 *
 * @param col (optional) the property tree
 *
 * @return float color from property tree
 */
fColor parseFColor(boost::optional<pt::ptree&> col) {
	if (col) {
		return fColor(
			col->get("<xmlattr>.r", 0.0f),
			col->get("<xmlattr>.g", 0.0f),
			col->get("<xmlattr>.b", 0.0f));
	}
	else {
		return fColor();
	}
};

/**
 * Parse colormap from property tree
 *
 * @param cmap (optional) the property tree
 *
 * @return colormap from property tree
 */
colormap parseColormap(boost::optional<pt::ptree&> cmap) {
	if (cmap) {
		if (boost::optional<std::string> preset = cmap->get_optional<std::string>("<xmlattr>.preset")) {
		
			// Parse preset
			return fromPreset(*preset);
		
		} else {
			
			// Parse types of colormaps
			std::string type = cmap->get("<xmlattr>.type", "mono"); // It's a real thing, just drink it.
			if (type == "gradient") {
				return colormap::gradient(
					parseColor(cmap->get_child_optional("from")),
					parseColor(cmap->get_child_optional("to")));
			}
			else if (type == "sinusoid") {
				return colormap::sinusoidWithAlpha(
					parseFColor(cmap->get_child_optional("frequency")),
					parseFColor(cmap->get_child_optional("phase")),
					cmap->get("<xmlattr>.alpha", 0xff));
			}
			else {
				return colormap();
			}
		
		}
	} else {
		return colormap();
	}
}

/**
 * Executes job described in property tree
 *
 * @param job the job tree
 */
void doFractalJob(pt::ptree job) {
	// Get values from xml job tree
	std::string mnemonic = job.get("<xmlattr>.mnemonic", "xml-fractal");
	bool mbrot = job.get("<xmlattr>.mandelbrot", false);
	cuFloatComplex cons = make_cuFloatComplex(
		job.get("constant.<xmlattr>.real", -0.4f), 
		job.get("constant.<xmlattr>.imag", 0.6f));
	cuFloatComplex scale = make_cuScaleComplex(
		job.get("scale.<xmlattr>.rotate", 0.0f),
		job.get("scale.<xmlattr>.zoom", 1.0f));
	cuFloatComplex trans = make_cuFloatComplex(
		job.get("translate.<xmlattr>.transx", 0.0f),
		job.get("translate.<xmlattr>.transy", 0.0f));
	unsigned width = job.get("image.<xmlattr>.width", 1920);
	unsigned height = job.get("image.<xmlattr>.height", 1080);
	std::string filename = job.get("image.<xmlattr>.filename", "fractal.png");
	colormap cmap = parseColormap(job.get_child_optional("colormap"));

	// Print job values in verbose
	VERBOSE("---------------JOB INFO---------------");
	VERBOSE("Mnemonic: " + mnemonic);
	VERBOSE("Mandelbrot: " + std::to_string(mbrot));
	VERBOSE("Constant: " + COMPLEX_STRING(cons));
	VERBOSE("Scale: " + COMPLEX_STRING(scale));
	VERBOSE("Translate: " + COMPLEX_STRING(trans));
	VERBOSE("Width: " + std::to_string(width));
	VERBOSE("Height: " + std::to_string(height));
	VERBOSE("File: " + filename);
	VERBOSE("--------------------------------------");

	// Generate fractal job
	generate(mbrot, cons, scale, trans, cmap, width, height, filename, mnemonic);
};

/**
 * Parses jobs from xml file with given filename
 *
 * @param filename the name of xml file
 */
void parseXmlFile(std::string filename) {
	VERBOSE("Parsing " + filename);
	std::ifstream file(filename);
	if (file.is_open())
	{
		VERBOSE("File is found!");
		try
		{
			// Parse jobs from file
			pt::ptree jobs;
			read_xml(file, jobs);
			VERBOSE("Valid XML");

			// Get joblist
			pt::ptree joblist = jobs.get_child("fractals");
			VERBOSE("Number of jobs: " + std::to_string(joblist.size()));

			// Do all jobs in fractals element
			BIG_DOING("Doing jobs...");
			BOOST_FOREACH(pt::ptree::value_type jobentry, joblist) {
				doFractalJob(jobentry.second);
			}
			BIG_DONE();
		}
		catch (const pt::xml_parser_error& e) 
		{
			std::cerr << "ERROR Parsing xml file!: " << e.what() << std::endl;
			file.close();
			return;
		}
		catch (const std::exception& e)
		{
			// Close file before throwing
			std::cerr << "ERROR: " << e.what() << std::endl;
			file.close();
			return;
		}

		// Close file
		file.close();
	}
	else
	{
		VERBOSE("File doesn't exist!");
	}
}

// -------------------------------- COMMAND PARSE ---------------------------------

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
	bool help, cmaps, mbrot;
	float consr, consi, zoom, rotate, transx, transy;
	unsigned width, height;
	std::string xml, cname, filename, mnemonic;

	// Get user input
	po::options_description options("> CUDAFractal [options]");
	options.add_options()
		("help", po::bool_switch(&help), "print help message")
		("cmaps", po::bool_switch(&cmaps), "prints the list of colormap presets")
		("xml", po::value<std::string>(&xml)->default_value(""), "parse xml file")
		("mbrot", po::bool_switch(&mbrot), "compute the mandelbrot fractal algorithm")
		("cr", po::value<float>(&consr)->default_value(-0.4), "real value of c")
		("ci", po::value<float>(&consi)->default_value(0.6), "imaginary value of c")
		("width", po::value<unsigned>(&width)->default_value(1920), "image width")
		("height", po::value<unsigned>(&height)->default_value(1080), "image height")
		("zoom", po::value<float>(&zoom)->default_value(1.0f), "zoom value")
		("rotate", po::value<float>(&rotate)->default_value(0.0f), "rotation value")
		("transx", po::value<float>(&transx)->default_value(0.0f), "x translation")
		("transy", po::value<float>(&transy)->default_value(0.0f), "y translation")
		("cmap", po::value<std::string>(&cname)->default_value("nvidia"), "colormap preset")
		("file", po::value<std::string>(&filename)->default_value("fractal.png"), "output file name")
		("mnemonic", po::value<std::string>(&mnemonic)->default_value("fractal"), "used to identify job")
		("verbose", po::bool_switch(&verbose), "print verbose messages");
	po::variables_map vars;
	po::store(po::parse_command_line(argc, argv, options), vars);
	po::notify(vars);

	// Handle different command options
	if (help) std::cout << options << std::endl;
	else if (!xml.empty()) parseXmlFile(xml);
	else if (cmaps) listPresets();
	else generate(mbrot, 
			make_cuFloatComplex(consr, consi), 
			make_cuScaleComplex(rotate, zoom), 
			make_cuFloatComplex(transx, transy), 
			fromPreset(cname), 
			width, height, filename, 
			mnemonic);

	return 0;
}

#endif // !__FRACTAL_HOST_CODE__