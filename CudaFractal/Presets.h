#pragma once

// Includes
#include "Coloring.cuh"
#include "Colormap.h"
#include <string>

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