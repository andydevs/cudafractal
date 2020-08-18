#include "XMLColormapParse.h"
#include "Presets.h"
#include <sstream>
#include <boost\property_tree\xml_parser.hpp>

/**
 * Parse hex value from string
 *
 * @param str the hex string
 *
 * @return hex value from string
 */
unsigned parseHex(std::string str) {
	unsigned val;
	std::stringstream stream(str);
	stream >> std::hex >> val;
	return val;
};

/**
 * Parse color from property tree
 *
 * @param col         (optional) the property tree
 * @param defaultHexa default hex color to return if color is not found
 */
rgba parseColor(boost::optional<pt::ptree&> col, std::string defaultHexa) {
	if (col) {
		std::string type = col->get("<xmlattr>.type", "rgba");
		if (type == "hex") {
			return rgbaFromHex(parseHex(col->get("<xmlattr>.hex", defaultHexa)));
		}
		else if (type == "hexa") {
			return rgbaFromHexa(parseHex(col->get("<xmlattr>.hexa", defaultHexa)));
		}
		else {
			rgba color;
			color.r = col->get("<xmlattr>.r", 0x00);
			color.g = col->get("<xmlattr>.g", 0x00);
			color.b = col->get("<xmlattr>.b", 0x00);
			color.a = col->get("<xmlattr>.a", 0xff);
			return color;
		}
	}
}

/**
 * Parse color from property tree
 *
 * @param col (optional) the property tree
 *
 * @return color from property tree
 */
color parseLegacyColor(boost::optional<pt::ptree&> col) {
	if (col) {
		// Parse different types of colors
		std::string type = col->get("<xmlattr>.type", "mono");
		if (type == "hex") {
			return color::hex(
				parseHex(
					col->get(
						"<xmlattr>.hex",
						"0x0000000")));
		}
		else if (type == "hexa") {
			return color::hexa(
				parseHex(
					col->get(
						"<xmlattr>.hexa",
						"0x0000000")));
		}
		else if (type == "rgb") {
			return color(
				col->get("<xmlattr>.r", 0x00),
				col->get("<xmlattr>.g", 0x00),
				col->get("<xmlattr>.b", 0x00));
		}
		else if (type == "rgba") {
			return color(
				col->get("<xmlattr>.r", 0x00),
				col->get("<xmlattr>.g", 0x00),
				col->get("<xmlattr>.b", 0x00),
				col->get("<xmlattr>.a", 0x00));
		}
		else {
			return color();
		}
	}
	else {
		return color();
	}
};

/**
 * Parse float color from property tree
 *
 * @param col (optional) the property tree
 *
 * @return float color from property tree
 */
fColor parseFColor(boost::optional<pt::ptree&> col) {
	if (col) {
		return fColor(
			col->get("<xmlattr>.r", 0.0f),
			col->get("<xmlattr>.g", 0.0f),
			col->get("<xmlattr>.b", 0.0f));
	}
	else {
		return fColor();
	}
};

/**
 * Parse colormap from property tree
 *
 * @param cmap (optional) the property tree
 *
 * @return colormap from property tree
 */
colormap_struct parseColormap(boost::optional<pt::ptree&> cmap) {
	if (cmap) {
		if (boost::optional<std::string> preset = cmap->get_optional<std::string>("<xmlattr>.preset")) {
			// Parse preset
			return fromPreset(*preset);
		}
		else {
			// Parse types of colormaps
			std::string type = cmap->get("<xmlattr>.type", "mono"); // It's a real thing, just drink it.
			if (type == "gradient") {
				return createGradient(
					parseColor(cmap->get_child_optional("from"), "0x000000ff"),
					parseColor(cmap->get_child_optional("to"), "0xffffffff"));
			}
			else if (type == "sinusoid") {
				return createLegacy(colormap::sinusoidWithAlpha(
					parseFColor(cmap->get_child_optional("frequency")),
					parseFColor(cmap->get_child_optional("phase")),
					cmap->get("<xmlattr>.alpha", 0xff)));
			}
			else {
				return createLegacy(colormap());
			}
		}
	}
	else {
		return createLegacy(colormap());
	}
}