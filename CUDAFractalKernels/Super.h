#pragma once

// PNG Image format
#define IMAGE_NUM_CHANNELS 4
#define IMAGE_RED_CHANNEL 0
#define IMAGE_GREEN_CHANNEL 1
#define IMAGE_BLUE_CHANNEL 2
#define IMAGE_ALPHA_CHANNEL 3

// Define max number of byte
#define BYTE_MAX 255

// Define byte type
typedef unsigned char byte;

// Define rgba struct
struct rgba { byte r, g, b, a; };

// Define API Attribute
#ifdef EXPORT
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif