_NOTE: README TO BE UPDATED_

# CudaFractal

Generates fractal images using Nvidia's CUDA framework.

![Julia Set (f(z) = z^2 - 0.4 + 0.6i)](flower.png)

## Usage

### Install

[REWRITE INSTALL]

### Command Line

The format is `> CudaFractal [options]` The options are as follows

|     Option     |                                               Description                                               |   Default   |
|:--------------:|:-------------------------------------------------------------------------------------------------------:|:-----------:|
| --help         | Print help message (overrides all other options)                                                        |             |
| --cmaps        | Print list of available colormap presets (overrides all other options except help)                      |             |
| --xml arg      | Generate fractal images from an xml file (see below. overrides all other options except help and cmaps) |             |
| --mbrot        | Generate mandelbrot image (overrides cr and ci options)                                                 |             |
| --cr arg       | Set real component of complex constant to arg                                                           |    -0.4     |
| --ci arg       | Set omaginary component of complex constant to arg                                                      |     0.6     |
| --width arg    | Set width of image to arg                                                                               |    1920     |
| --height arg   | Set height of image to arg                                                                              |    1080     |
| --zoom arg     | Zoom image by arg                                                                                       |     1.0     |
| --rotate arg   | Rotate image by arg degrees                                                                             |     0.0     |
| --transx arg   | Translate image horizontally by arg                                                                     |     0.0     |
| --transy arg   | Translate image vertically by arg                                                                       |     0.0     |
| --cmap arg     | Set colormap preset being used to arg                                                                   |   nvidia    |
| --file arg     | Set filename being saved to to arg                                                                      | fractal.png |
| --mnemonic arg | Set mnemonic to arg (best used by scripts to identify what fractal is being made)                       | fractal     |
| --verbose      | Print verbose messages in program                                                                       |             |

### XML File

Fractals can be specified in an xml file, and can be read in the program using `> CudaFractal --xml [xml file name]`

The XML file format is as follows:

```
<?xml ...?>
<fractals>
	
	[fractal specs...]

<fractals/>
```

A list of fractal specs within the "fractals" tag.

Each spec has the following format:

```
<fractal mnemonic="[mnemonic of fractal (default fractal)]" mandelbrot="[true if mandelbrot set is generated (default false)]">
	<constant real="[real value (default -0.4)]" imag="[imaginary value (default 0.6)]"/> <!-- Ignored if mandelbrot is true -->
	<scale rotate="[rotation amount (default 0.0)]" zoom="[zoom amount (default 1.0)]"/>
	<translate transx="[horizontal translation (default 0.0)]" transy="[vertical translation (default 0.0)]"/>
	<image width="[image width (default 1920)]" height="[image height (default 1080)]" filename="[name of file (default fractal.png)]"/>
	
	[colormap spec...]
</fractal>
```

#### Colormaps

Colormaps can be specified in three ways. The default colormap is greyscale black to white.

##### Presets

Colormaps can be defined by preset as follows:

```
<colormap preset="[preset]"/>
```

##### Gradient

Gradient colormaps can be defined as follows:

```
<colormap type="gradient">
	<from [from color]/>
	<to [to color]/>
</colormap>
```

(see Colors section below)

##### Sinusoid

Sinusoid colormaps can be defined as follows

```
<colormap type="sinusoid" alpha="255">
	<frequency r="0.1" g="1.0" b="0.0"/>
	<phase r="1.0" g="2.0" b="0.1"/>
</colormap>
```

`alpha` defaults to `255`. `r`, `g`, `b`, values default to `0.0`

##### Colors

Colors can be defined by either using a `hex` (or `hexa` with alpha channel) value

```
<color type="hex" hex="0x00aaff" />
<color type="hexa" hexa="0x00ffaaff"/>
```

`hex` and `hexa` values default to `0x0000000`

Or by the individual `rgb` (or `rgba` with alpha channel) values

```
<color type="rgb" r="0" g="124" b="109"/>
<color type="rgba" r="0" g="20" b="200" a="255"/>
```

`r`, `g`, `b`, and `a` values default to `0`
