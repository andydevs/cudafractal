# CudaFractal

Generates fractal images using Nvidia's CUDA framework.

![Flower JuliaSet Image](flower.png)

## Usage

### Install

Requirements: Windows 10 (or later) 64-bit, Nvidia GPU

Download InstallCudaFractal.msi from the release page and run it. This will install the program into C:\Program Files\Andydevs\CudaFractal.

#### Optional: Add C:\Program Files\Andydevs\CudaFractal To Your Path

Normally, you would have to enter `> C:\Program Files\Andydevs\CudaFractal\CudaFractal` to run CudaFractal. To shorten this to `> CudaFractal`, follow these steps:

- Open `Control Panel` and navigate to `System and Security > System`
- Click `Change Settings`
- Go to the `Advanced` tab and click `Environment Variables`
- Locate `Path` in `System variables`. Click on it, and then click `Edit` below.
- Click `New`, and then add `C:\Program Files\Andydevs\CudaFractal`
- Click `Ok`

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