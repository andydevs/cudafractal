#ifndef __FRACTAL_DEVICE_CODE__
#define __FRACTAL_DEVICE_CODE__

// Includes
#include "Coloring.cuh"
#include "Super.cuh"
#include <cuComplex.h>

/**
 * It assigns the corresponding pixel of the thread to a corresponding starting
 * complex number z. Then, it runs the juliaset algorithm on z using the given c.
 * Finally, it computes the color from the resulting iteration number and assigns
 * that color to the thread's corresponding pixel in the image.
 *
 * @param c    the complex constant c
 * @param s    the scale complex
 * @param t    the translation complex
 * @param cmap the colormap to use when mapping colors
 * @param w    the width of the image
 * @param h    the height of the image
 * @param img  the image buffer
 */
__global__
void juliaset(cuFloatComplex c, cuFloatComplex s, cuFloatComplex t, colormap cmap, unsigned w, unsigned h, unsigned char* img);

/**
 * It assigns the corresponding pixel of the thread to a corresponding complex
 * constant number c and sets z to 0. Then, it runs the iteration algorithm on
 * z using the given c.Finally, it computes the color from the resulting iteration
 * number and assigns that color to the thread's corresponding pixel in the image.
 *
 * @param s the scale complex
 * @param t the translation complex
 * @param cmap the colormap to use when mapping colors
 * @param w    the width of the image
 * @param h    the height of the image
 * @param img  the image buffer
 */
__global__
void mandelbrotset(cuFloatComplex s, cuFloatComplex t, colormap cmap, unsigned w, unsigned h, unsigned char* img);

#endif // !__FRACTAL_DEVICE_CODE__
