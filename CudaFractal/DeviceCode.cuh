#ifndef __FRACTAL_DEVICE_CODE__
#define __FRACTAL_DEVICE_CODE__

// Includes
#include <cuComplex.h>

// PNG Image format
#define IMAGE_NUM_CHANNELS 4
#define IMAGE_RED_CHANNEL 0
#define IMAGE_GREEN_CHANNEL 1
#define IMAGE_BLUE_CHANNEL 2
#define IMAGE_ALPHA_CHANNEL 3

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
