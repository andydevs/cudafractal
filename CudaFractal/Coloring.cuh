#ifndef __FRACTAL_COLORING__
#define __FRACTAL_COLORING__

// Includes
#include "Super.cuh"

// --------------------------------------- COLOR STRUCT ---------------------------------------

// Color struct
struct color { 
	byte r, g, b, a; 

	/**
	 * Empty constructor
	 */
	__device__ __host__
	color() : 
		r(0), g(0), b(0), a(0xff) {};

	/**
	 * Copy constructor
	 *
	 * @param other other color
	 */
	__device__ __host__
	color(const color& other) : 
		r(other.r), g(other.g), b(other.b), a(other.a) {};

	/**
	 * Constructor for rgba values
	 *
	 * @param r the r channel
	 * @param g the g channel
	 * @param b the b channel
	 * @param a the a channel
	 */
	__device__ __host__
	color(byte r, byte g, byte b, byte a) :
		r(r), g(g), b(b), a(a) {};

	/**
	* Constructor for rgb values
	*
	* @param r the r channel
	* @param g the g channel
	* @param b the b channel
	*/
	__device__ __host__
	color(byte r, byte g, byte b) :
		r(r), g(g), b(b), a(0xff) {};

	/**
	 * Creates a color from the given hex value
	 *
	 * @param val the hex value
	 *
	 * @return color from hex value
	 */
	__device__ __host__
	static color hexa(unsigned val) {
		return color(
			(val & 0xff000000) >> 24,
			(val & 0x00ff0000) >> 16,
			(val & 0x0000ff00) >> 8,
			(val & 0x000000ff));
	};

	/**
	 * Creates a color from the given hex value (alpha is default 0xff)
	 *
	 * @param val the hex value
	 *
	 * @return color from hex value
	 */
	__device__ __host__
	static color hex(unsigned val) {
		return color(
			(val & 0xff0000) >> 16,
			(val & 0x00ff00) >> 8,
			(val & 0x0000ff));
	};
};

// ------------------------------------ HELPER FUNCTIONS ------------------------------------

/**
 * A gradient value between the given from
 * and to values using iter value (generated
 * from number of iterations)
 *
 * @param iter the iter value
 * @param from the starting value
 * @param to   the ending value
 *
 * @return resulting gradient value
 */
__device__ __host__ __inline__
byte byteGrad(byte iter, byte from, byte to) {
	if (to == from) return from;
	else if (to > from) return from + (to - from) * iter / BYTE_MAX;
	else return from - (from - to) * iter / BYTE_MAX;
}

/**
 * Gradient map between two colors
 *
 * @param iter the iter value
 * @param from the starting color
 * @param to   the ending color
 *
 * @return resulting gradient color
 */
__device__ __host__ __inline__
color gradient(byte iter, color from, color to) {
	return color(
		byteGrad(iter, from.r, to.r),
		byteGrad(iter, from.g, to.g),
		byteGrad(iter, from.b, to.b),
		byteGrad(iter, from.a, to.a));
}

// -------------------------------------- COLORMAPS ---------------------------------------

// Colormap type
enum colormap_type {
	GRADIENT
};

// Colormap struct
struct colormap {
	// Data
	colormap_type type;
	color from;
	color to;

	/**
	 * Empty constructor
	 */
	__device__ __host__
	colormap() {};

	/**
	 * Copy constructor
	 *
	 * @param other the other colormap to copy
	 */
	__device__ __host__
	colormap(const colormap& other) :
		type(other.type),
		from(other.from),
		to(other.to) {};

	/**
	 * Creates a new gradient colormap with the given from and to colors
	 *
	 * @param from the start color
	 * @param to the end color
	 */
	__device__ __host__
	colormap(color from, color to) :
		type(GRADIENT),
		from(from),
		to(to) {};
	
	/**
	 * Creates a new gradient colormap with the given from and to values
	 *
	 * @param from the start color
	 * @param to the end color
	 *
	 * @return new gradient colormap
	 */
	__device__ __host__
	static colormap gradient(color from, color to) {
		return colormap(from, to);
	};
};

/**
 * Maps the given iter value to a color according to the given colormap
 *
 * @param cmap the colormap being used
 * @param iter the iterations value
 *
 * @return color mapped by the given iter value
 */
__device__ __host__ __inline__
color map(colormap cmap, byte iter) {
	switch (cmap.type)
	{
		case GRADIENT: return gradient(iter, cmap.from, cmap.to);
		// Standard black to white colormap
		default: return color(iter, iter, iter);
	}
};

// -------------------------------------- SET PIXEL ----------------------------------------

/**
 * Sets the pixel in the image to the given color
 *
 * @param img the image
 * @param w the width of the image
 * @param h the height of the image
 * @param x the x coord of the pixel
 * @param y the y coord of the pixel
 */
__device__ __host__ __inline__
void setPixel(byte* img, unsigned w, unsigned h, unsigned x, unsigned y, color col) {
	img[(y*w + x)*IMAGE_NUM_CHANNELS + IMAGE_RED_CHANNEL]   = col.r; // Red
	img[(y*w + x)*IMAGE_NUM_CHANNELS + IMAGE_GREEN_CHANNEL] = col.g; // Green
	img[(y*w + x)*IMAGE_NUM_CHANNELS + IMAGE_BLUE_CHANNEL]  = col.b; // Blue
	img[(y*w + x)*IMAGE_NUM_CHANNELS + IMAGE_ALPHA_CHANNEL] = col.a; // Alpha
}

#endif