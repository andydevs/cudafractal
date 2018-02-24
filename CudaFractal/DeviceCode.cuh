#ifndef __FRACTAL_DEVICE_CODE__
#define __FRACTAL_DEVICE_CODE__

// Includes
#include "Super.cuh"
#include <cuComplex.h>

/**
* It assigns the corresponding pixel of the thread to a corresponding starting
* complex number z. Then, it runs the juliaset algorithm on z using the given c.
* Finally, it computes the color from the resulting iteration number and assigns
* that color to the thread's corresponding pixel in the image.
*
* @param c   the complex constant c
* @param w   the width of the image
* @param h   the height of the image
* @param img the image buffer
*/
__global__
void juliaset(cuFloatComplex c, unsigned w, unsigned h, unsigned char* img);

#endif // !__FRACTAL_DEVICE_CODE__
