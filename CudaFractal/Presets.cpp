#include "Coloring.cuh"
#include "Colormap.h"
#include "Super.h"
#include "Presets.h"
#include "XMLParse.h"
 
#include <iostream>
#include <map>

// Presets map
static bool uninitialized = true;
static std::map<std::string, colormap_struct> presets;

/**
 * Return preset file path
 *
 * @return preset file path
 */
std::string getPresetFilePath() {
#ifdef _DEBUG
	return ".\\presets.xml";
#else
	return getenv("CUDAFractalFiles") + std::string("\\presets.xml");
#endif
}

/**
 * Initializes the presets map from preset file
 */
void initPresets() {
	std::string presetFile = getPresetFilePath();
	parsePresetXmlFile(presetFile, presets);
}

/**
 * DEPRECATED
 * Directly initializes the presets map
 */
void initPresets_legacy() {
	// Initialize if uninitialized
	if (uninitialized) {
		// Populate presets map
		VERBOSE("Initialize presets map");
		presets["noir"] = createGradient(
			rgbaFromHex(0x000000), 
			rgbaFromHex(0xffffff));
		presets["ink"] = createGradient(
			rgbaFromHex(0xffffff), 
			rgbaFromHex(0x000000));
		presets["nvidia"] = createGradient(
			rgbaFromHex(0x000000), 
			rgbaFromHex(0xa3ff00));
		presets["orchid"] = createGradient(
			rgbaFromHex(0xeeeeff), 
			rgbaFromHex(0xff0000));
		presets["flower"] = createLegacy(colormap::sinusoid(
			fColor(0.7, 0.7, 0.7),
			fColor(-2.0, -2.0, -1.0)));
		presets["psychedelic"] = createLegacy(colormap::sinusoid(
			fColor(5.0, 5.0, 5.0),
			fColor(4.1, 4.5, 5.0)));
		presets["ice"] = createLegacy(colormap::sinusoid(
			fColor(2.0, 2.0, 0.1),
			fColor(0.0, 0.0, 2.0)));
		presets["fruity"] = createLegacy(colormap::sinusoid(
			fColor(5.0, 5.0, 5.0),
			fColor(0.0, 4.5, 2.5)));
		presets["sarree"] = createLegacy(colormap::sinusoid(
			fColor(1.4, 1.4, 1.4),
			fColor(2.0, 3.0, 4.0)));
		presets["saffron"] = createLegacy(colormap::sinusoid(
			fColor(1.00, 2.00, 2.00),
			fColor(F_P1, F_P1, F_P1)));
		presets["lightgarden"] = createLegacy(colormap::sinusoid(
			fColor(1.00, 2.00, 9.00),
			fColor(F_N1, F_N1, F_N1)));
		presets["acid"] = createLegacy(colormap::sinusoid(
			fColor(8.00, 9.00, 0.00),
			fColor(F_N1, F_N1, F_N1)));

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
colormap_struct fromPreset(std::string name) {
	return presets[name];
};

/**
 * Lists all presets available
 */
void listPresets() {
	std::cout << "Presets Available:" << std::endl;
	for each (std::pair<std::string, colormap_struct> entry in presets) {
		std::cout << "    " << entry.first;
		switch (entry.second.type)
		{
		case GRADIENT_TYPE:
			std::cout << " -- GRADIENT" << std::endl;
			break;
		case LEGACY_TYPE:
			std::cout << " -- LEGACY" << std::endl;
			break; // YOU WILL NOT FOOL ME TWICE
		default:
			break;
		}
	}
};