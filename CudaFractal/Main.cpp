// Includes
#include "Presets.h"
#include "XMLParse.h"
#include "Generate.h"
#include <cuda_runtime.h>

// Boost
#include <boost\optional.hpp>
#include <boost\program_options.hpp>

// Libraries
#include <iostream>
#include <string>

// Namespaces
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
	bool help, cmaps, mbrot;
	float consr, consi, zoom, rotate, transx, transy;
	unsigned width, height;
	std::string xml, cname, filename, mnemonic;

	// Initialize presets table
	initPresets();

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
		("mnemonic", po::value<std::string>(&mnemonic)->default_value("fractal"), "used to identify job");
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
		fromPreset(cname), width, height, filename,
		mnemonic);

	return 0;
}