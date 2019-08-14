// Inlcudes
#include "Coloring.cuh"
#include "Super.h"
#include "Presets.h"
#include "XMLParse.h"
#include "Generate.h"

// CUDA
#include <cuda_runtime.h>
#include <cuComplex.h>

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
#include <cstdio>
#include <cmath>
#include <ctime>
#include <map>

// Boost namespaces
namespace po = boost::program_options;
namespace pt = boost::property_tree;

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
		}
		else {
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
	}
	else {
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
	DEFINE_TIMES
		
	VERBOSE("Parsing " + filename);
	std::ifstream file(filename);
	if (file.is_open())
	{
		VERBOSE("File is found!");
		try
		{
			// Parse jobs from file
			pt::ptree jobs;
			read_xml(file, jobs, pt::xml_parser::no_comments);
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
