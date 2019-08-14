#include "Presets.cuh"
#include "Super.cuh"
#include "Coloring.cuh"
 
#include <iostream>
#include <map>

// Presets map
static bool uninitialized = true;
static std::map<std::string, colormap> presets;

/**
 * Initializes the presets map
 */
void initPresets() {
	// Initialize if uninitialized
	if (uninitialized) {
		// Populate presets map
		VERBOSE("Initialize presets map");
		presets["noir"] = colormap::gradient(
			color::hex(0x000000),
			color::hex(0xffffff));
		presets["ink"] = colormap::gradient(
			color::hex(0xffffff),
			color::hex(0x000000));
		presets["nvidia"] = colormap::gradient(
			color::hex(0x000000),
			color::hex(0xa3ff00));
		presets["orchid"] = colormap::gradient(
			color::hex(0xeeeeff),
			color::hex(0xff0000));
		presets["flower"] = colormap::sinusoid(
			fColor(0.7, 0.7, 0.7),
			fColor(-2.0, -2.0, -1.0));
		presets["psychedelic"] = colormap::sinusoid(
			fColor(5.0, 5.0, 5.0),
			fColor(4.1, 4.5, 5.0));
		presets["ice"] = colormap::sinusoid(
			fColor(2.0, 2.0, 0.1),
			fColor(0.0, 0.0, 2.0));
		presets["fruity"] = colormap::sinusoid(
			fColor(5.0, 5.0, 5.0),
			fColor(0.0, 4.5, 2.5));
		presets["sarree"] = colormap::sinusoid(
			fColor(1.4, 1.4, 1.4),
			fColor(2.0, 3.0, 4.0));
		presets["saffron"] = colormap::sinusoid(
			fColor(1.00, 2.00, 2.00),
			fColor(F_P1, F_P1, F_P1));
		presets["lightgarden"] = colormap::sinusoid(
			fColor(1.00, 2.00, 9.00),
			fColor(F_N1, F_N1, F_N1));
		presets["acid"] = colormap::sinusoid(
			fColor(8.00, 9.00, 0.00),
			fColor(F_N1, F_N1, F_N1));

		// Toggle uninitialized
		uninitialized = false;
	}
};

/**
 * Returns the preset colormap of the given name
 *
 * @param name the name of the colormap
 *
 * @return the preset colormap
 */
colormap fromPreset(std::string name) {
	initPresets();
	return presets[name];
};

/**
 * Lists all presets available
 */
void listPresets() {
	initPresets();
	std::cout << "Presets Available:" << std::endl;
	for each (std::pair<std::string, colormap> entry in presets) {
		std::cout << "    " << entry.first << std::endl;
	}
};