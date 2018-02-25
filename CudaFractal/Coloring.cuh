#ifndef __FRACTAL_COLORING__
#define __FRACTAL_COLORING__

// Includes
#include "Super.cuh"

// Color struct
typedef struct { byte r, g, b, a; } color;

/**
 * Creates a color from the given rgba values
 *
 * @param r the red channel of the color
 * @param g the green channel of the color
 * @param b the blue channel of the color
 * @param a the alpha channel of the color
 *
 * @return color from given rgba values
 */
__device__ __host__ __inline__
color rgba(byte r, byte g, byte b, byte a) {
	color out;
	out.r = r;
	out.g = g;
	out.b = b;
	out.a = a;
	return out;
}

/**
* Creates a color from the given rgb values (a is default 0xff)
*
* @param r the red channel of the color
* @param g the green channel of the color
* @param b the blue channel of the color
*
* @return color from given rgb values
*/
__device__ __host__ __inline__
color rgb(byte r, byte g, byte b) {
	color out;
	out.r = r;
	out.g = g;
	out.b = b;
	out.a = 0xff;
	return out;
}

/**
* Creates a color from the given hex value
*
* @param val the hex value
*
* @return color from hex value
*/
__device__ __host__ __inline__
color hexa(unsigned val) {
	return rgba(
		(val & 0xff000000) >> 24,
		(val & 0x00ff0000) >> 16,
		(val & 0x0000ff00) >> 8,
		(val & 0x000000ff));
}

/**
 * Creates a color from the given hex value (alpha is default 0xff)
 *
 * @param val the hex value
 *
 * @return color from hex value
 */
__device__ __host__ __inline__
color hex(unsigned val) {
	return rgb(
		(val & 0xff0000) >> 16,
		(val & 0x00ff00) >> 8,
		(val & 0x0000ff));
}

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
byte gradient(byte iter, byte from, byte to) {
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
color colorGrad(byte iter, color from, color to) {
	color out;
	out.r = gradient(iter, from.r, to.r);
	out.g = gradient(iter, from.g, to.g);
	out.b = gradient(iter, from.b, to.b);
	out.a = gradient(iter, from.a, to.a);
	return out;
}

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