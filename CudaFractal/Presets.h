#pragma once

// Includes
#include "Coloring.cuh"
#include "Colormap.h"
#include <string>

/**
 * Return preset file path
 *
 * @return preset file path
 */
std::string getPresetFilePath();

/**
 * Initializes the presets map
 */
void initPresets();

/**
 * Returns the preset colormap of the given name
 *
 * @param name the name of the colormap
 *
 * @return the preset colormap
 */
colormap_struct fromPreset(std::string name);

/**
 * Lists all presets available
 */
void listPresets();