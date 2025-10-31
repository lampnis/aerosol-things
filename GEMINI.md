# Key Context Files for Our Collaboration

All the following files are located in `additional-context/`!
- **`GEMINI.md` (This file):** My primary instructions, project overview, and our agreed-upon working style.
- **`WORKLOG.md`:** A detailed, summarized log of the project's technical progress, generated from `notes.txt`.
- **`SESSIONS.md`:** A high-level log of our chat sessions and what we accomplished in each one.
- **`TREE.md`:** A snapshot of the project's directory structure, updated at the beginning and end of each session.
In the beginning of each session, check these files and `git log`, to get up to speed

To update the `WORKLOG.md`, look in the root directory of the project for a file:
- **`notes.txt`:** Your raw, handwritten notes that serve as the source for `WORKLOG.md`.

## 0. Startup Routine

At the beginning of each session, you should perform the following steps to get up to speed and ensure consistency:

1.  **Update Project Tree:** Generate the latest project directory tree and save it to `additional-context/TREE.md`.
    ```bash
    ls -R /home/labuser/aerosol-things > /home/labuser/aerosol-things/additional-context/TREE.md
    ```
2.  **Review READMEs:** Read all `README.md` files in the project to understand the context of different directories.
    ```bash
    find /home/labuser/aerosol-things -name "README.md" -exec cat {} +
    ```
3.  **Review Worklog:** Read the `additional-context/WORKLOG.md` to catch up on the latest technical progress.
    ```bash
    cat /home/labuser/aerosol-things/additional-context/WORKLOG.md
    ```
4.  **Review Session Log:** Read the `additional-context/SESSIONS.md` to understand the high-level accomplishments of previous sessions.
    ```bash
    cat /home/labuser/aerosol-things/additional-context/SESSIONS.md
    ```
5.  **Review Git Log:** Check the recent commit history for any changes not yet reflected in the logs.
    ```bash
    git log -n 5
    ```
6.  **Compare and Identify Discrepancies:** Manually compare the generated `TREE.md` and the contents of `README.md` files with the `WORKLOG.md` and `SESSIONS.md` to identify any discrepancies or areas that need updating. This helps in understanding the current state of the project and what was accomplished.

At the end of each session (if no crash occurred), you should update `additional-context/TREE.md` again to reflect any changes made during the session.
    ```bash
    ls -R /home/labuser/aerosol-things > /home/labuser/aerosol-things/additional-context/TREE.md
    ```

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
