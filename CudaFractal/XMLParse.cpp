// Inlcudes
#include "Super.h"
#include "Presets.h"
#include "XMLColormapParse.h"
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
#include <cstdio>
#include <cmath>
#include <ctime>
#include <map>

// Boost namespaces
namespace po = boost::program_options;
namespace pt = boost::property_tree;

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
	
	// Create legacy colormap
	colormap_struct cmap = parseColormap(job.get_child_optional("colormap"));

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
