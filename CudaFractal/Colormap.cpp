#include "Colormap.h"

colormap_struct createLegacy(colormap cmap) {
	colormap_struct cmaps;
	cmaps.type = LEGACY_TYPE;
	cmaps.legacy_map = cmap;
	return cmaps;
};

rgba rgbaFromHexa(int hexa) {
	rgba color;
	color.r = (0xff000000 & hexa) >> 32;
	color.g = (0x00ff0000 & hexa) >> 16;
	color.b = (0x0000ff00 & hexa) >> 8;
	color.a = (0x000000ff & hexa) >> 0;
	return color;
};

rgba rgbaFromHex(int hex) {
	rgba color;
	color.r = (0xff0000 & hex) >> 16;
	color.g = (0x00ff00 & hex) >> 8;
	color.b = (0x0000ff & hex) >> 0;
	color.a = 0xff;
	return color;
}

colormap_struct createGradient(rgba from, rgba to) {
	colormap_struct cmap;
	cmap.type = GRADIENT_TYPE;
	cmap.from = from;
	cmap.to = to;
	return cmap;
};