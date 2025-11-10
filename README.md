# 🔬 Aerosol Data Analysis Project 📊

This repository is dedicated to the comprehensive analysis of experimental data derived from aerosol measurements. Our primary objective is to deeply understand the intricate relationships between various experimental parameters, optimizing our setup and analysis workflows for precision and insight.

## ✨ Project Overview

Our core mission revolves around analyzing data from a sophisticated, custom-built experimental setup. The bulk of our analytical work is performed within intuitive Jupyter notebooks, with all data meticulously organized in a structured directory format.

Key aspects of this project include:
-   **Data Acquisition 📈**: Utilizing a custom LabView program for precise data collection, integrating hardware like the PicoScope AWG for laser current control.
-   **Spectral Analysis 🧪**: In-depth examination of spectral data across diverse experimental conditions, including varying pressures, temperatures, and laser modulation frequencies.
-   **Hardware Comparison 🛠️**: Evaluating the performance of different hardware configurations, such as various cantilevers and Helmholtz acoustic filters, to optimize the experimental setup.
-   **Noise & SNR Optimization 📉**: Rigorous analysis of noise sources and continuous efforts to enhance the signal-to-noise ratio for more reliable measurements.
-   **Flow Control Dynamics 💧**: Investigation into flow rate dynamics within the CEPAS cell, leveraging newly integrated needle and electronic valves.

## 📁 Directory Structure at a Glance

Navigating this project is straightforward with our organized structure:

-   `codes/datas/` 🗄️: The central archive for all raw and processed experimental data, meticulously categorized by experiment type and date. **All data here is treated as read-only to ensure integrity.**
-   `my_utils/` 🐍: Our bespoke Python package, brimming with helper functions, classes, and shared definitions crucial for data loading, elegant plotting, and robust analysis.
    -   `common.py`: Essential utilities for data handling and file operations.
    -   `classes.py`: Custom classes for structured data representation and complex analysis workflows.
    -   `defs.py`: Project-wide constants, AI integration functions, and specialized spectroscopic tools.
-   `additional-context/` 📚: A dedicated space for project documentation, including worklogs, session summaries, and directory tree snapshots, essential for seamless collaboration.
-   `*.ipynb` 📝: Jupyter notebooks, serving as our primary interface for interactive data analysis, exploration, and compelling visualization.
-   `*.opju` (under `opjs/`) 📊: OriginPro project files for specialized data visualization and analysis tasks.

## 🏷️ Data Naming Convention

Files within `codes/datas/` strictly adhere to a consistent naming pattern to ensure clarity and easy identification:

`[measurement_type]_[pressure]_[signal_amplitude]_[modulation_frequency]__msr__[run_number]`

Example: `gasx_300_38_20__msr__0`
-   `gasx`: Type of measurement (e.g., 'gasx' for spectral sweeps).
-   `300`: Pressure in mbar.
-   `38`: Signal Amplitude in V.
-   `20`: Laser modulation frequency in Hz.
-   `__msr__`: A mandatory "measurement" identifier.
-   `0`: Repetition or run number for unique identification.

## 🚀 Getting Started

Dive into the data analysis by exploring the Jupyter notebooks located in the root directory. The `my_utils` package provides all the necessary functions and classes to effortlessly load, process, and visualize the experimental data.

To begin, we recommend exploring `aislop.ipynb` or `algebra.ipynb` to see the analysis in action!