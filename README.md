# CudaFractal

Generates fractal images using Nvidia's CUDA framework.

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

| Option  | Description        | Default |
|:-------:|:------------------:|:-------:|
| --help  | Print help message |         |

### XML File