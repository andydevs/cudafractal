#pragma once

// Verbose flag
#define verbose 1

/**
 * Returns scale complex which incorporates rotation and zooming
 *
 * @param rotate the rotation value (in degrees)
 * @param zoom   the zoom value
 *
 * @return scale complex
 */
#define make_cuScaleComplex(rotate, zoom) \
	make_cuFloatComplex( \
		cos(rotate*F_PI / 180.0f) / zoom, \
		sin(rotate*F_PI / 180.0f) / zoom)

// Debug Macros
#define DEFINE_TIMES \
	clock_t big_start; \
	clock_t start;
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