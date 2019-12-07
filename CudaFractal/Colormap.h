#pragma once
#include "Super.h"
#include "Coloring.cuh"

enum colormap_struct_type {
	GRADIENT_TYPE,
	LEGACY_TYPE
};

struct colormap_struct {
	colormap_struct_type type;
	rgba from;
	rgba to;
	colormap legacy_map;
};

colormap_struct createLegacy(colormap cmap);

rgba rgbaFromHexa(int hexa);

rgba rgbaFromHex(int hex);

colormap_struct createGradient(rgba from, rgba to);