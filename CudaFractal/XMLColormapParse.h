#pragma once
#include "Colormap.h"
#include <string>
#include <boost\optional.hpp>
#include <boost\property_tree\ptree.hpp>

namespace pt = boost::property_tree;

/**
 * Parse hex value from string
 *
 * @param str the hex string
 *
 * @return hex value from string
 */
unsigned parseHex(std::string str);

rgba parseColor(boost::optional<pt::ptree&> col, std::string defaultHexa);

/**
 * Parse color from property tree
 *
 * @param col (optional) the property tree
 *
 * @return color from property tree
 */
color parseLegacyColor(boost::optional<pt::ptree&> col);

/**
 * Parse float color from property tree
 *
 * @param col (optional) the property tree
 *
 * @return float color from property tree
 */
fColor parseFColor(boost::optional<pt::ptree&> col);

/**
 * Parse colormap from property tree
 *
 * @param cmap (optional) the property tree
 *
 * @return colormap from property tree
 */
colormap_struct parseColormap(boost::optional<pt::ptree&> cmap);