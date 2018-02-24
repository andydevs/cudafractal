#ifndef __FRACTAL_COLORING__
#define __FRACTAL_COLORING__

// Includes
#include "Super.cuh"

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

#endif