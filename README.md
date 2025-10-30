# Aerosol Data Analysis Project

This repository contains the code and data for a series of experiments on aerosol measurements. The project involves analyzing experimental data to understand the relationships between different experimental parameters.

## Project Overview

The primary goal of this project is to analyze data from a custom-built experimental setup. The analysis is primarily conducted in Jupyter notebooks, and the data is stored in a structured directory format.

The project involves:
-   Data acquisition from a custom LabView program.
-   Analysis of spectral data from different experimental conditions (pressure, temperature, modulation frequency, etc.).
-   Comparison of different hardware configurations (e.g., cantilevers).
-   Noise analysis and signal-to-noise ratio (SNR) optimization.

## Directory Structure

-   `codes/datas/`: Contains the raw and processed experimental data, organized by experiment type and date.
-   `my_utils/`: A Python package with helper functions for data loading, plotting, and analysis.
-   `*.ipynb`: Jupyter notebooks for data analysis and visualization.

## Data Naming Convention

The data files in `codes/datas/` follow a specific naming convention:

`gasx_300_38_20__msr__0`

-   `gasx`: Type of measurement.
-   `300`: Pressure in mbar.
-   `38`: Signal Amplitude in V.
-   `20`: Laser modulation frequency.
-   `__msr__`: "measurement" identifier.
-   `0`: Repetition or run number.

## Getting Started

To get started with the analysis, you can explore the Jupyter notebooks in the root directory. The `my_utils` package contains the necessary functions to load and process the data.