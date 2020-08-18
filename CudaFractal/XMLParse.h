#pragma once

// Include
#include "Colormap.h"
#include <string>
#include <map>

/**
 * Parses jobs from xml file with given filename
 *
 * @param filename the name of xml file
 */
void parseXmlFile(std::string filename);

/**
 * Read preset xml file into colormap preset map
 *
 * @param filename name of preset xml file
 * @param presets  presets map to set to
 */
void parsePresetXmlFile(std::string filename, std::map<std::string, colormap_struct>& presets);