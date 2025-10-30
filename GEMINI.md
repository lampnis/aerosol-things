# Key Context Files for Our Collaboration

- **`GEMINI.md` (This file):** My primary instructions, project overview, and our agreed-upon working style.
- **`WORKLOG.md`:** A detailed, summarized log of the project's technical progress, generated from `notes.txt`.
- **`SESSIONS.md`:** A high-level log of our chat sessions and what we accomplished in each one.
- **`notes.txt`:** Your raw, handwritten notes that serve as the source for `WORKLOG.md`.
In the beginning of each session, check these files and `git log`, to get up to speed
---

# GEMINI Project Context: Aerosol Data Analysis

This document provides context for the Gemini-CLI assistant to effectively help with this project.

## 1. Project Overview

This project is for analyzing experimental data related to aerosol measurements. The primary activities involve data exploration, analysis, and visualization, mostly conducted within Jupyter notebooks (`.ipynb` files). The goal is to understand the relationships between different experimental parameters.

## 2. Directory Structure & Key Files

-   `/` (root): Contains the primary analysis notebooks (`aislop.ipynb`, `algebra.ipynb`, etc.). These are the main entry points for analysis.
-   `codes/datas/`: This is the central repository for all raw and processed experimental data. The directory structure is organized by experiment type and date. **This data should be treated as read-only.**
-   `my_utils/`: A Python package containing helper functions, classes, and common definitions used across the different notebooks to avoid code duplication.
    -   `common.py`: Utility functions for data loading and plotting.
    -   `classes.py`: Custom classes for data representation.
    -   `defs.py`: Project-wide constants and definitions.

## 3. Data Schema & Naming Convention

The filenames in the `codes/datas/` directory follow a specific pattern. Understanding this is key to finding and interpreting data.

**Example:** `gasx_300_38_20__msr__0`

Please fill in the meaning of each part:
-   `gasx`: [e.g., Type of measurement ('gasx', 'open' etc.)]
        For now the used types are:
            'single': A single point (fixed laser frequency)
                measurement over time, used as one of the 'noise'
                measures;
            'gasx': most common, spectral sweep over multiple frequencies;
            'open': can be either of two previous ones, but mainly used as
                    a label to know that that is measurement for the cell that
                    will be used in open-cell setting eventually
-   `_300_`: [e.g., Pressure in mbar]
-   `_38_`: [e.g., Signal Amplitude in V (not sure if mV or uV)]
-   `_20_`: [e.g., Laser modulation frequency]
-   `__msr__`: [e.g., "measurement" identifier]
-   `_0`: [e.g., Repetition or run number]

## 4. Working Style & Conventions
-   **My Role**: My primary role is to be an assistant for data analysis, exploration, and organization. I can help with finding files, generating code snippets for plotting or analysis, and answering questions about the data based on the schema you provide.
-   **Code Edits**: You prefer that I **do not** directly edit code files (`.py`) or notebooks (`.ipynb`) unless explicitly asked. Instead, provide suggestions, code snippets, and commands that you can run.
-   **Data Integrity**: Data files under `codes/datas/` are immutable and should never be modified.
-   **Commands**: When you need to run something, provide the `python` or `jupyter` command. Assume a standard scientific Python environment (numpy, pandas, matplotlib) is available.
-   **Organization**: Help with organizing and writing out some README.md files in each directory and also root directory, for better navigation and understandability of the project.
-   **AI help**: Help with better integration of AI tools (agents, MCPs etc.) in tandem with Gemini-CLI

## 5. Current Task / Scratchpad

Use this space to keep track of the current analysis goal.

-   **Current Goal**: [e.g., "Investigate the effect of pressure on signal amplitude for the '22deg' dataset."]
