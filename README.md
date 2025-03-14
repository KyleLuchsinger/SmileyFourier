# Fourier Series Smiley Face Animation

<img src="./src/smiley_fourier.gif" alt="Fourier Series Smiley Face Animation" width="400" />

## Overview

This project creates an animated visualization of a smiley face using Fourier series. The animation demonstrates how complex shapes can be approximated using epicycles (circles rotating on circles), which is a visual representation of Fourier transform principles.

## Features

- Generates a detailed smiley face outline with eyes, pupils, eyebrows, and smile
- Computes Fourier coefficients to represent the curve mathematically
- Creates an animation showing how epicycles recreate the original outline
- Includes progress tracking during computation and rendering
- Produces a high-quality GIF animation as output

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- SciPy
- tqdm

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/KyleLuchsinger/SmileyFourier.git
cd SmileyFourier
pip install numpy matplotlib scipy tqdm
```

## Usage

Run the main script to generate the animation:

```bash
python SF.py
```

The script will:
1. Generate a smiley face outline
2. Compute the Fourier coefficients
3. Create an animation showing how epicycles trace the outline
4. Save the animation as `smiley_fourier.gif`

The process may take several minutes depending on your computer's specifications.

## How it Works

The Fourier series representation allows any periodic function to be expressed as a sum of sine and cosine terms. In this context, a closed curve (the smiley face) is treated as a periodic function.

The main steps of the algorithm are:

1. **Generate Shape**: Create a detailed outline of a smiley face using parametric equations
2. **Compute Coefficients**: Calculate the Fourier coefficients that represent the shape
3. **Visualize Epicycles**: Animate the drawing process using a series of rotating circles (epicycles)

For each frame of the animation, the position of each epicycle is calculated based on:
- The magnitude of its corresponding Fourier coefficient (determines the radius)
- The frequency (determines the rotation speed)
- The phase (determines the starting angle)

## Customization

You can modify the following parameters in the code:

- `num_coeffs` in `compute_fourier_coefficients()`: Controls the number of Fourier coefficients (higher means more detail)
- `num_terms` in `animate_fourier_drawing()`: Sets how many epicycles to show in the animation
- `duration_sec` and `fps` in `animate_fourier_drawing()`: Control the animation length and frame rate

## Mathematical Background

The Fourier series representation of a complex-valued function can be written as:

$$f(t) = \sum_{k=-\infty}^{\infty} c_k e^{ikt}$$

Where the coefficients $c_k$ are calculated by:

$$c_k = \frac{1}{2\pi} \int_{0}^{2\pi} f(t) e^{-ikt} dt$$

In this application, the x and y coordinates of the shape are combined into a complex number z = x + iy, and the Fourier coefficients are computed for this complex function.
