#pragma once

// Includes
#include "Super.h"
#include "Coloring.cuh"
#include <cuComplex.h>

/**
 * Launch juliaset kernel
 *
 * It assigns the corresponding pixel of the thread to a corresponding starting
 * complex number z. Then, it runs the juliaset algorithm on z using the given c
 * and saves the result to the space
 *
 * @param c   the complex constant c
 * @param s   the scale complex
 * @param t   the translation complex
 * @param w   the width of the image
 * @param h   the height of the image
 * @param spc the image buffer
 */
void juliaset_launcher(cuFloatComplex c, cuFloatComplex s, cuFloatComplex t, unsigned w, unsigned h, byte *spc);

/**
 * Launch mandelbrotset kernel
 *
 * It assigns the corresponding pixel of the thread to a corresponding complex
 * constant number c and sets z to 0. Then, it runs the iteration algorithm on
 * z using the given c and saves the result to the space buffer
 *
 * @param s   the scale complex
 * @param t   the translation complex
 * @param w   the width of the image
 * @param h   the height of the image
 * @param spc the image buffer
 */
void mandelbrotset_launcher(cuFloatComplex s, cuFloatComplex t, unsigned w, unsigned h, byte *spc);
