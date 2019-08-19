#pragma once
#include "Coloring.cuh"

/**
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