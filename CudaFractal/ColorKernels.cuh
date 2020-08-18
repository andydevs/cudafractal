#pragma once
#include "Coloring.cuh"

/**
 * DEPRECATED (gonna use static colormap array)
 * 
 * Launch legacy colormap kernel
 *
 * Use legacy colormap code to map number from iteration space to byte color in image
 *
 * @param cmap  colormap struct used
 * @param w     width of image
 * @param h     height of image
 * @param space fractal iteration space buffer
 * @param image final image buffer
 */
void legacy_colormap_launcher(colormap cmap, unsigned w, unsigned h, byte *space, byte *image);

/**
 * DEPRECATED (gonna use static colormap array)
 *
 * Launch gradient colormap kernel
 *
 * Map number from iteration space to byte color in image from linear gradient between two colors
 *
 * @param from  initial color in gradient
 * @param to    final color in gradient
 * @param w     width of image
 * @param h     height of image
 * @param space fractal iteration space buffer
 * @param image final image buffer
 */
void gradient_colormap_launcher(rgba from, rgba to, unsigned w, unsigned h, byte *space, byte *image);